from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
EVENTS_PATH = BASE_DIR / "events.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DASHBOARD_PATH = BASE_DIR / "dashboard.html"

RANDOM_SEED = 42
EVENT_CHUNK_SIZE = 500_000


ORDER_COLUMNS = [
    "order_id",
    "user_id",
    "status",
    "gender",
    "created_at",
    "returned_at",
    "shipped_at",
    "delivered_at",
    "num_of_item",
    "product_id",
    "sale_price",
    "age",
    "state",
    "city",
    "country",
    "traffic_source",
    "cost",
    "category",
    "brand",
    "retail_price",
    "department",
    "distribution_center_id",
    "product_name",
    "warehouse_name",
    "is_loyal",
    "customer_review",
]

EVENT_COLUMNS = [
    "user_id",
    "session_id",
    "created_at",
    "browser",
    "traffic_source",
    "uri",
    "event_type",
]


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def clean_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    replacements = {
        "Â®": "®",
        "Â©": "©",
        "â€™": "'",
        "â€“": "-",
        "â€”": "-",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def qscore(series: pd.Series, reverse: bool = False) -> pd.Series:
    filled = series.fillna(series.median())
    ranks = filled.rank(method="first", pct=True).fillna(0.5)
    if reverse:
        ranks = 1 - ranks
    return np.clip(np.ceil(ranks * 5), 1, 5).astype(int)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(0)


def read_orders() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, usecols=lambda col: col in ORDER_COLUMNS, low_memory=False)
    for col in ["created_at", "returned_at", "shipped_at", "delivered_at"]:
        if col in data.columns:
            data[col] = parse_datetime(data[col])

    for col in ["product_name", "customer_review", "category", "brand", "department"]:
        if col in data.columns:
            data[col] = data[col].map(clean_text)

    data["sale_price"] = pd.to_numeric(data["sale_price"], errors="coerce").fillna(0.0)
    data["cost"] = pd.to_numeric(data["cost"], errors="coerce").fillna(0.0)
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["margin"] = data["sale_price"] - data["cost"]
    data["is_returned"] = data.get("returned_at", pd.Series(index=data.index)).notna().astype(int)
    data["is_cancelled"] = data["status"].eq("Cancelled").astype(int)
    data["is_completed"] = data["status"].eq("Complete").astype(int)
    data["is_shipped"] = data["status"].eq("Shipped").astype(int)
    data["shipping_days"] = (data["shipped_at"] - data["created_at"]).dt.total_seconds() / 86400
    data["delivery_days"] = (data["delivered_at"] - data["created_at"]).dt.total_seconds() / 86400
    return data


def build_order_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_date = data["created_at"].max() + pd.Timedelta(days=1)

    order_level = (
        data.sort_values("created_at")
        .drop_duplicates(["order_id", "user_id"], keep="last")
        .copy()
    )

    customer = data.groupby("user_id").agg(
        items=("product_id", "size"),
        total_spend=("sale_price", "sum"),
        total_margin=("margin", "sum"),
        unique_products=("product_id", "nunique"),
        unique_categories=("category", "nunique"),
        avg_item_price=("sale_price", "mean"),
        returned_items=("is_returned", "sum"),
        cancelled_items=("is_cancelled", "sum"),
        completed_items=("is_completed", "sum"),
        shipped_items=("is_shipped", "sum"),
        avg_shipping_days=("shipping_days", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        age=("age", "median"),
        gender=("gender", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        country=("country", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        city=("city", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        traffic_source=("traffic_source", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        is_loyal=("is_loyal", "max"),
        first_order=("created_at", "min"),
        last_order=("created_at", "max"),
    )

    order_counts = order_level.groupby("user_id").agg(
        orders=("order_id", "nunique"),
        cancelled_orders=("is_cancelled", "sum"),
        completed_orders=("is_completed", "sum"),
    )
    customer = customer.join(order_counts, how="left")
    customer["avg_order_value"] = safe_divide(customer["total_spend"], customer["orders"])
    customer["return_rate"] = safe_divide(customer["returned_items"], customer["items"])
    customer["cancel_rate"] = safe_divide(customer["cancelled_items"], customer["items"])
    customer["completion_rate"] = safe_divide(customer["completed_items"], customer["items"])
    customer["recency_days"] = (reference_date - customer["last_order"]).dt.days
    customer["tenure_days"] = (customer["last_order"] - customer["first_order"]).dt.days.clip(lower=0)
    customer["purchase_frequency"] = safe_divide(customer["orders"], customer["tenure_days"] + 1)
    customer["rfm_r"] = qscore(customer["recency_days"], reverse=True)
    customer["rfm_f"] = qscore(customer["orders"])
    customer["rfm_m"] = qscore(customer["total_spend"])
    customer["rfm_score"] = customer["rfm_r"] * 100 + customer["rfm_f"] * 10 + customer["rfm_m"]

    revenue_rank = customer["total_spend"].sort_values(ascending=False)
    cumulative_share = revenue_rank.cumsum() / max(revenue_rank.sum(), 1)
    abc = pd.Series("C", index=revenue_rank.index)
    abc.loc[cumulative_share <= 0.80] = "A"
    abc.loc[(cumulative_share > 0.80) & (cumulative_share <= 0.95)] = "B"
    customer["abc_segment"] = abc.reindex(customer.index).fillna("C")

    monthly = (
        order_level.assign(month=order_level["created_at"].dt.tz_convert(None).dt.to_period("M").astype(str))
        .groupby(["user_id", "month"])
        .size()
        .rename("monthly_orders")
        .reset_index()
    )
    xyz_stats = monthly.groupby("user_id")["monthly_orders"].agg(["mean", "std"]).fillna(0)
    coeff_var = safe_divide(xyz_stats["std"], xyz_stats["mean"])
    customer["xyz_segment"] = np.select(
        [coeff_var <= 0.50, coeff_var <= 1.00],
        ["X", "Y"],
        default="Z",
    )

    category_sales = data.groupby(["category", "department"]).agg(
        revenue=("sale_price", "sum"),
        margin=("margin", "sum"),
        items=("product_id", "size"),
        return_rate=("is_returned", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_price=("sale_price", "mean"),
    ).reset_index()
    category_sales["quality_risk"] = (
        category_sales["return_rate"] * 0.55 + category_sales["cancel_rate"] * 0.45
    )

    product_quality = data.groupby(["product_id", "product_name", "category", "brand"]).agg(
        revenue=("sale_price", "sum"),
        items=("product_id", "size"),
        return_rate=("is_returned", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    product_quality["quality_risk"] = (
        product_quality["return_rate"] * 0.55 + product_quality["cancel_rate"] * 0.45
    )

    return customer, category_sales, product_quality


def aggregate_events() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(EVENTS_PATH, usecols=lambda col: col in EVENT_COLUMNS, chunksize=EVENT_CHUNK_SIZE, low_memory=False):
        chunk["user_id"] = pd.to_numeric(chunk["user_id"], errors="coerce")
        chunk = chunk.dropna(subset=["user_id"]).copy()
        if chunk.empty:
            continue
        chunk["user_id"] = chunk["user_id"].astype("int64")
        chunk["created_at"] = parse_datetime(chunk["created_at"])

        base = chunk.groupby("user_id").agg(
            event_count=("event_type", "size"),
            sessions=("session_id", "nunique"),
            first_event=("created_at", "min"),
            last_event=("created_at", "max"),
        )

        event_counts = pd.crosstab(chunk["user_id"], chunk["event_type"])
        event_counts = event_counts.add_prefix("event_")
        parts.append(base.join(event_counts, how="left").fillna(0))

    if not parts:
        return pd.DataFrame()

    events = pd.concat(parts).reset_index()
    sum_cols = [col for col in events.columns if col not in {"user_id", "first_event", "last_event"}]
    aggregated = events.groupby("user_id").agg(
        {**{col: "sum" for col in sum_cols}, "first_event": "min", "last_event": "max"}
    )
    aggregated["events_per_session"] = safe_divide(aggregated["event_count"], aggregated["sessions"])
    for col in ["event_cart", "event_purchase", "event_cancel", "event_product", "event_department"]:
        if col not in aggregated:
            aggregated[col] = 0
    aggregated["cart_to_purchase_rate"] = safe_divide(aggregated["event_purchase"], aggregated["event_cart"])
    aggregated["cancel_event_rate"] = safe_divide(aggregated["event_cancel"], aggregated["event_count"])
    return aggregated


def merge_features(order_features: pd.DataFrame, event_features: pd.DataFrame) -> pd.DataFrame:
    features = order_features.join(event_features, how="left")
    numeric_event_cols = event_features.select_dtypes(include=[np.number]).columns if not event_features.empty else []
    for col in numeric_event_cols:
        features[col] = features[col].fillna(0)
    return features


def make_churn_target(features: pd.DataFrame) -> pd.Series:
    high_recency = features["recency_days"] > features["recency_days"].quantile(0.75)
    weak_orders = features["orders"] <= features["orders"].quantile(0.35)
    high_cancel = features["cancel_rate"] > max(0.25, features["cancel_rate"].quantile(0.80))
    high_return = features["return_rate"] > max(0.12, features["return_rate"].quantile(0.80))
    low_engagement = features.get("events_per_session", pd.Series(0, index=features.index)) <= features.get("events_per_session", pd.Series(0, index=features.index)).quantile(0.30)
    return ((high_recency & weak_orders) | high_cancel | high_return | (high_recency & low_engagement)).astype(int)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = positives.sum()
    n_neg = negatives.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train_churn_model(features: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    model_cols = [
        "items",
        "orders",
        "total_spend",
        "total_margin",
        "avg_order_value",
        "unique_products",
        "unique_categories",
        "return_rate",
        "cancel_rate",
        "completion_rate",
        "avg_shipping_days",
        "avg_delivery_days",
        "recency_days",
        "tenure_days",
        "purchase_frequency",
        "rfm_r",
        "rfm_f",
        "rfm_m",
        "event_count",
        "sessions",
        "events_per_session",
        "cart_to_purchase_rate",
        "cancel_event_rate",
    ]
    model_cols = [col for col in model_cols if col in features.columns]
    X_df = features[model_cols].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    y = make_churn_target(features).to_numpy(dtype=float)

    X = X_df.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    means = np.nan_to_num(X.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    stds = np.nan_to_num(X.std(axis=0), nan=1.0, posinf=1.0, neginf=1.0)
    stds[stds == 0] = 1
    Xs = np.nan_to_num(np.clip((X - means) / stds, -8, 8), nan=0.0, posinf=8.0, neginf=-8.0)
    Xb = np.column_stack([np.ones(Xs.shape[0]), Xs])

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(Xb.shape[0])
    rng.shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, test_idx = indices[:split], indices[split:]

    weights = np.zeros(Xb.shape[1])
    learning_rate = 0.02
    l2 = 0.001
    for _ in range(650):
        weights = np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0)
        weights = np.clip(weights, -20, 20)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            logits = np.nan_to_num(Xb[train_idx] @ weights, nan=0.0, posinf=30.0, neginf=-30.0)
            logits = np.clip(logits, -30, 30)
            preds = 1 / (1 + np.exp(-logits))
            gradient = (Xb[train_idx].T @ (preds - y[train_idx])) / len(train_idx)
        gradient[1:] += l2 * weights[1:]
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=5.0, neginf=-5.0)
        gradient = np.clip(gradient, -5, 5)
        weights = np.clip(weights - learning_rate * gradient, -20, 20)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        final_logits = np.nan_to_num(Xb @ weights, nan=0.0, posinf=30.0, neginf=-30.0)
        probabilities = 1 / (1 + np.exp(-np.clip(final_logits, -30, 30)))
    test_probabilities = probabilities[test_idx]
    test_y = y[test_idx]
    accuracy = float(((test_probabilities >= 0.5).astype(int) == test_y).mean())
    auc = auc_score(test_y, test_probabilities)

    predictions = pd.DataFrame(
        {
            "user_id": features.index,
            "churn_probability": probabilities,
            "churn_label": y.astype(int),
            "risk_group": pd.cut(
                probabilities,
                bins=[-0.01, 0.35, 0.65, 1.01],
                labels=["low", "medium", "high"],
            ).astype(str),
        }
    )

    coefficients = pd.DataFrame(
        {
            "feature": model_cols,
            "coefficient": weights[1:],
            "importance": np.abs(weights[1:]),
        }
    ).sort_values("importance", ascending=False)

    metrics = {
        "model": "numpy_logistic_regression",
        "target_definition": "synthetic churn proxy: high recency with weak activity, high cancellation, high return, or low engagement",
        "rows": int(len(features)),
        "positive_rate": float(y.mean()),
        "accuracy": accuracy,
        "auc": auc,
        "features": model_cols,
    }
    return predictions, metrics, coefficients


def add_customer_segments(features: pd.DataFrame) -> pd.Series:
    score = (
        features["rfm_r"].astype(str)
        + features["rfm_f"].astype(str)
        + features["rfm_m"].astype(str)
    )
    conditions = [
        (features["abc_segment"].eq("A") & (features["rfm_f"] >= 4) & (features["rfm_m"] >= 4)),
        (features["rfm_r"] <= 2) & (features["rfm_m"] >= 4),
        (features["rfm_f"] >= 4) & (features["rfm_m"] <= 3),
        (features["cancel_rate"] > 0.25) | (features["return_rate"] > 0.15),
        (features["orders"] <= 1),
    ]
    labels = [
        "champions",
        "valuable_sleeping",
        "frequent_value_seekers",
        "service_risk",
        "new_or_one_time",
    ]
    return pd.Series(np.select(conditions, labels, default="regular"), index=features.index) + "_" + score


def build_recommendations(
    data: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    product_quality: pd.DataFrame,
) -> pd.DataFrame:
    quality_threshold = product_quality["quality_risk"].quantile(0.75)
    good_products = product_quality[
        (product_quality["quality_risk"] <= quality_threshold)
        & (product_quality["items"] >= max(3, product_quality["items"].quantile(0.10)))
    ].copy()
    if good_products.empty:
        good_products = product_quality.copy()

    popularity = good_products.sort_values(
        ["quality_risk", "revenue", "avg_margin"],
        ascending=[True, False, False],
    )

    user_categories = data.groupby("user_id")["category"].apply(lambda x: set(x.dropna().astype(str)))
    prediction_index = predictions.set_index("user_id")
    risky_users = prediction_index.sort_values("churn_probability", ascending=False).head(5000)

    rows = []
    top_by_category = {
        cat: grp.head(5)
        for cat, grp in popularity.groupby("category", sort=False)
    }
    global_top = popularity.head(30)

    for user_id, pred in risky_users.iterrows():
        seen = user_categories.get(user_id, set())
        bought_categories = list(seen)
        candidates = []
        for cat in bought_categories:
            if cat in top_by_category:
                candidates.append(top_by_category[cat])
        candidates.append(global_top[~global_top["category"].isin(seen)].head(10))
        candidate_df = pd.concat(candidates).drop_duplicates("product_id").head(3)
        for rank, row in enumerate(candidate_df.itertuples(index=False), start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "rank": rank,
                    "churn_probability": float(pred["churn_probability"]),
                    "recommended_product_id": int(row.product_id),
                    "recommended_product": row.product_name,
                    "category": row.category,
                    "brand": row.brand,
                    "reason": "retention_offer_low_quality_risk",
                    "quality_risk": float(row.quality_risk),
                }
            )

    return pd.DataFrame(rows)


def forecast_sales(data: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        data.assign(month=data["created_at"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(revenue=("sale_price", "sum"), orders=("order_id", "nunique"), items=("product_id", "size"))
        .sort_index()
    )
    if monthly.empty:
        return pd.DataFrame()

    last_month = monthly.index.max()
    last_values = monthly["revenue"].tail(6)
    base = last_values.tail(3).mean()
    trend = last_values.diff().tail(3).mean()
    if pd.isna(trend):
        trend = 0.0

    forecasts = []
    for step in range(1, 4):
        month = last_month + pd.DateOffset(months=step)
        predicted = max(float(base + trend * step), 0.0)
        forecasts.append(
            {
                "month": month.strftime("%Y-%m"),
                "forecast_revenue": predicted,
                "method": "last_3_month_average_plus_short_trend",
            }
        )
    return pd.DataFrame(forecasts)


def compact_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    abs_value = abs(float(value))
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"


def svg_bar_chart(items: pd.DataFrame, label_col: str, value_col: str, width: int = 760, height: int = 280) -> str:
    if items.empty:
        return "<p>No data</p>"
    items = items.head(10).copy()
    max_value = max(float(items[value_col].max()), 1.0)
    row_height = height / len(items)
    bars = []
    for idx, row in enumerate(items.itertuples(index=False)):
        label = str(getattr(row, label_col))[:34]
        value = float(getattr(row, value_col))
        bar_width = (width - 260) * value / max_value
        y = idx * row_height + 8
        bars.append(
            f'<text x="0" y="{y + 16:.1f}" font-size="12">{escape(label)}</text>'
            f'<rect x="250" y="{y:.1f}" width="{bar_width:.1f}" height="18" fill="#247a7b"></rect>'
            f'<text x="{255 + bar_width:.1f}" y="{y + 14:.1f}" font-size="12">{escape(compact_number(value))}</text>'
        )
    return f'<svg viewBox="0 0 {width} {height}" role="img">{"".join(bars)}</svg>'


def table_html(df: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if df.empty:
        return "<p>No data</p>"
    rows = []
    for row in df[columns].head(limit).itertuples(index=False):
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in row)
        rows.append(f"<tr>{cells}</tr>")
    headers = "".join(f"<th>{escape(col)}</th>" for col in columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def generate_dashboard(
    data: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    category_sales: pd.DataFrame,
    recommendations: pd.DataFrame,
    sales_forecast: pd.DataFrame,
    metrics: dict,
) -> None:
    risk_counts = predictions["risk_group"].value_counts().rename_axis("risk_group").reset_index(name="customers")
    top_categories = category_sales.sort_values("revenue", ascending=False)
    risky_customers = predictions.sort_values("churn_probability", ascending=False).head(10)
    feature_importance = coefficients.head(10)
    segments = features["customer_segment"].value_counts().head(10).rename_axis("segment").reset_index(name="customers")

    revenue = data["sale_price"].sum()
    orders = data["order_id"].nunique()
    customers = data["user_id"].nunique()
    high_risk_share = (predictions["risk_group"] == "high").mean()

    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Customer analytics dashboard</title>
  <style>
    :root {{
      --ink: #182022;
      --muted: #5f6f73;
      --line: #d9e1e3;
      --bg: #f7f9f9;
      --panel: #ffffff;
      --accent: #247a7b;
      --accent-2: #b64b5d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.45;
    }}
    header, section {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px;
    }}
    header {{
      padding-top: 36px;
      border-bottom: 1px solid var(--line);
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric strong {{
      display: block;
      font-size: 26px;
      margin-top: 6px;
    }}
    .two {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    svg {{ width: 100%; height: auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 9px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 700; }}
    .note {{ font-size: 13px; }}
  </style>
</head>
<body>
  <header>
    <h1>Комплексная клиентская аналитика</h1>
    <p>От диагностики продаж и поведения до прогноза риска оттока, удерживающих рекомендаций и мониторинга качества товарных предложений.</p>
  </header>

  <section class="grid">
    <div class="metric">Выручка<strong>{compact_number(revenue)}</strong></div>
    <div class="metric">Заказы<strong>{orders:,}</strong></div>
    <div class="metric">Клиенты<strong>{customers:,}</strong></div>
    <div class="metric">Высокий риск оттока<strong>{high_risk_share:.1%}</strong></div>
  </section>

  <section class="two">
    <div class="panel">
      <h2>Топ категорий по выручке</h2>
      {svg_bar_chart(top_categories, "category", "revenue")}
    </div>
    <div class="panel">
      <h2>Группы риска</h2>
      {svg_bar_chart(risk_counts, "risk_group", "customers")}
    </div>
  </section>

  <section class="two">
    <div class="panel">
      <h2>Важные факторы оттока</h2>
      {table_html(feature_importance, ["feature", "coefficient", "importance"])}
      <p class="note">Модель: {escape(metrics["model"])}. Accuracy: {metrics["accuracy"]:.3f}; AUC: {metrics["auc"]:.3f}.</p>
    </div>
    <div class="panel">
      <h2>Сегменты клиентов</h2>
      {table_html(segments, ["segment", "customers"])}
    </div>
  </section>

  <section>
    <h2>Клиенты с максимальным риском</h2>
    {table_html(risky_customers, ["user_id", "churn_probability", "risk_group", "churn_label"])}
  </section>

  <section class="two">
    <div class="panel">
      <h2>Прогноз продаж</h2>
      {table_html(sales_forecast, ["month", "forecast_revenue", "method"])}
    </div>
    <div class="panel">
      <h2>Примеры удерживающих рекомендаций</h2>
      {table_html(recommendations, ["user_id", "rank", "recommended_product", "category", "reason"], limit=8)}
    </div>
  </section>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def write_artifacts(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    category_sales: pd.DataFrame,
    product_quality: pd.DataFrame,
    recommendations: pd.DataFrame,
    sales_forecast: pd.DataFrame,
    metrics: dict,
) -> None:
    features.reset_index().to_csv(ARTIFACTS_DIR / "customer_features.csv", index=False)
    predictions.to_csv(ARTIFACTS_DIR / "churn_predictions.csv", index=False)
    coefficients.to_csv(ARTIFACTS_DIR / "churn_feature_importance.csv", index=False)
    category_sales.to_csv(ARTIFACTS_DIR / "category_quality.csv", index=False)
    product_quality.to_csv(ARTIFACTS_DIR / "product_quality.csv", index=False)
    recommendations.to_csv(ARTIFACTS_DIR / "recommendations.csv", index=False)
    sales_forecast.to_csv(ARTIFACTS_DIR / "sales_forecast.csv", index=False)
    (ARTIFACTS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    data = read_orders()
    order_features, category_sales, product_quality = build_order_features(data)
    event_features = aggregate_events()
    features = merge_features(order_features, event_features)
    features["customer_segment"] = add_customer_segments(features)
    predictions, metrics, coefficients = train_churn_model(features)
    recommendations = build_recommendations(data, features, predictions, product_quality)
    sales_forecast = forecast_sales(data)
    write_artifacts(
        features,
        predictions,
        coefficients,
        category_sales,
        product_quality,
        recommendations,
        sales_forecast,
        metrics,
    )
    generate_dashboard(
        data,
        features,
        predictions,
        coefficients,
        category_sales,
        recommendations,
        sales_forecast,
        metrics,
    )
    print("Done")
    print(f"Dashboard: {DASHBOARD_PATH}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
