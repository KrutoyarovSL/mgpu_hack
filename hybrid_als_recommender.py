from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from cmfrec import CMF_implicit


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "hybrid_als"

RANDOM_STATE = 42
TOP_N = 10
RECOMMEND_USERS = 5_000
MAX_EVAL_USERS = 2_000
MAX_ONE_HOT_LEVELS = 25


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
    "inventory_item_id",
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
]


def ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def minmax(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    low = series.min()
    high = series.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series(0.0, index=series.index)
    return (series - low) / (high - low)


def one_hot_top(frame: pd.DataFrame, column: str, prefix: str, max_levels: int = MAX_ONE_HOT_LEVELS) -> pd.DataFrame:
    values = frame[column].fillna("unknown").astype(str)
    top_values = values.value_counts().head(max_levels).index
    clipped = values.where(values.isin(top_values), "other")
    encoded = pd.get_dummies(clipped, prefix=prefix, dtype=np.float32)
    return encoded


def read_orders() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, usecols=lambda col: col in ORDER_COLUMNS, low_memory=False)
    data["created_at"] = parse_datetime(data["created_at"])
    data["returned_at"] = parse_datetime(data["returned_at"])
    data["shipped_at"] = parse_datetime(data["shipped_at"])
    data["delivered_at"] = parse_datetime(data["delivered_at"])
    data["sale_price"] = pd.to_numeric(data["sale_price"], errors="coerce").fillna(0.0)
    data["cost"] = pd.to_numeric(data["cost"], errors="coerce").fillna(0.0)
    data["retail_price"] = pd.to_numeric(data["retail_price"], errors="coerce").fillna(data["sale_price"])
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["is_returned"] = data["returned_at"].notna().astype(int)
    data["is_cancelled"] = data["status"].eq("Cancelled").astype(int)
    data["is_completed"] = data["status"].eq("Complete").astype(int)
    data["margin"] = data["sale_price"] - data["cost"]
    data["shipping_days"] = (data["shipped_at"] - data["created_at"]).dt.total_seconds() / 86400
    data["delivery_days"] = (data["delivered_at"] - data["created_at"]).dt.total_seconds() / 86400
    return data.dropna(subset=["user_id", "product_id", "created_at"]).copy()


def build_user_info(data: pd.DataFrame) -> pd.DataFrame:
    reference_date = data["created_at"].max() + pd.Timedelta(days=1)
    order_level = data.sort_values("created_at").drop_duplicates(["user_id", "order_id"], keep="last")

    user_info = data.groupby("user_id").agg(
        user_items=("product_id", "size"),
        user_unique_items=("product_id", "nunique"),
        user_unique_categories=("category", "nunique"),
        user_total_spend=("sale_price", "sum"),
        user_total_margin=("margin", "sum"),
        user_avg_price=("sale_price", "mean"),
        user_return_rate=("is_returned", "mean"),
        user_cancel_rate=("is_cancelled", "mean"),
        user_completion_rate=("is_completed", "mean"),
        user_avg_shipping_days=("shipping_days", "mean"),
        user_avg_delivery_days=("delivery_days", "mean"),
        user_age=("age", "median"),
        user_first_order=("created_at", "min"),
        user_last_order=("created_at", "max"),
        gender=("gender", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        country=("country", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        traffic_source=("traffic_source", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        is_loyal=("is_loyal", "max"),
    ).reset_index()

    order_counts = order_level.groupby("user_id").agg(user_orders=("order_id", "nunique")).reset_index()
    user_info = user_info.merge(order_counts, on="user_id", how="left")
    user_info["user_recency_days"] = (reference_date - user_info["user_last_order"]).dt.days
    user_info["user_tenure_days"] = (user_info["user_last_order"] - user_info["user_first_order"]).dt.days.clip(lower=0)
    user_info["user_avg_order_value"] = safe_divide(user_info["user_total_spend"], user_info["user_orders"])
    user_info["user_purchase_frequency"] = safe_divide(user_info["user_orders"], user_info["user_tenure_days"] + 1)

    numeric_cols = [
        "user_items",
        "user_unique_items",
        "user_unique_categories",
        "user_total_spend",
        "user_total_margin",
        "user_avg_price",
        "user_return_rate",
        "user_cancel_rate",
        "user_completion_rate",
        "user_avg_shipping_days",
        "user_avg_delivery_days",
        "user_age",
        "user_orders",
        "user_recency_days",
        "user_tenure_days",
        "user_avg_order_value",
        "user_purchase_frequency",
    ]
    numeric = user_info[numeric_cols].apply(minmax).astype(np.float32)
    encoded = pd.concat(
        [
            one_hot_top(user_info, "gender", "gender", 5),
            one_hot_top(user_info, "country", "country", MAX_ONE_HOT_LEVELS),
            one_hot_top(user_info, "traffic_source", "traffic", MAX_ONE_HOT_LEVELS),
        ],
        axis=1,
    )
    side = pd.concat([user_info[["user_id"]].rename(columns={"user_id": "UserId"}), numeric, encoded], axis=1)
    return side


def build_item_info(data: pd.DataFrame) -> pd.DataFrame:
    item_info = data.groupby("product_id").agg(
        item_sales=("product_id", "size"),
        item_buyers=("user_id", "nunique"),
        item_revenue=("sale_price", "sum"),
        item_margin=("margin", "sum"),
        item_avg_price=("sale_price", "mean"),
        item_retail_price=("retail_price", "mean"),
        item_return_rate=("is_returned", "mean"),
        item_cancel_rate=("is_cancelled", "mean"),
        item_completion_rate=("is_completed", "mean"),
        category=("category", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        brand=("brand", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        department=("department", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
        product_name=("product_name", lambda x: x.mode().iat[0] if not x.mode().empty else "unknown"),
    ).reset_index()
    item_info["item_quality_risk"] = item_info["item_return_rate"] * 0.55 + item_info["item_cancel_rate"] * 0.45
    item_info["item_margin_rate"] = safe_divide(item_info["item_margin"], item_info["item_revenue"])

    numeric_cols = [
        "item_sales",
        "item_buyers",
        "item_revenue",
        "item_margin",
        "item_avg_price",
        "item_retail_price",
        "item_return_rate",
        "item_cancel_rate",
        "item_completion_rate",
        "item_quality_risk",
        "item_margin_rate",
    ]
    numeric = item_info[numeric_cols].apply(minmax).astype(np.float32)
    encoded = pd.concat(
        [
            one_hot_top(item_info, "category", "category", MAX_ONE_HOT_LEVELS),
            one_hot_top(item_info, "brand", "brand", MAX_ONE_HOT_LEVELS),
            one_hot_top(item_info, "department", "department", MAX_ONE_HOT_LEVELS),
        ],
        axis=1,
    )
    side = pd.concat([item_info[["product_id"]].rename(columns={"product_id": "ItemId"}), numeric, encoded], axis=1)
    lookup = item_info[["product_id", "product_name", "category", "brand", "item_quality_risk"]].rename(
        columns={"product_id": "ItemId"}
    )
    return side, lookup


def build_user_item(data: pd.DataFrame) -> pd.DataFrame:
    interactions = data.groupby(["user_id", "product_id"]).agg(
        times_bought=("product_id", "size"),
        total_spend=("sale_price", "sum"),
        completed=("is_completed", "sum"),
        returned=("is_returned", "sum"),
        cancelled=("is_cancelled", "sum"),
        last_seen=("created_at", "max"),
    ).reset_index()

    spend_score = minmax(np.log1p(interactions["total_spend"]))
    count_score = np.log1p(interactions["times_bought"])
    quality_penalty = 0.35 * interactions["returned"] + 0.45 * interactions["cancelled"]
    interactions["Value"] = (1.0 + count_score + 0.75 * spend_score + 0.15 * interactions["completed"] - quality_penalty)
    interactions["Value"] = interactions["Value"].clip(lower=0.1).astype(np.float32)
    return interactions.rename(columns={"user_id": "UserId", "product_id": "ItemId"})[
        ["UserId", "ItemId", "Value", "times_bought", "total_spend", "last_seen"]
    ]


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_dates = data.groupby(["user_id", "order_id"])["created_at"].max().reset_index()
    order_dates["order_rank_desc"] = order_dates.groupby("user_id")["created_at"].rank(method="first", ascending=False)
    latest_orders = order_dates[order_dates["order_rank_desc"].eq(1)][["user_id", "order_id"]]
    order_counts = order_dates.groupby("user_id")["order_id"].nunique()
    eligible_users = set(order_counts[order_counts >= 2].index)
    latest_orders = latest_orders[latest_orders["user_id"].isin(eligible_users)]

    split_keys = latest_orders.assign(is_test=True)
    marked = data.merge(split_keys, on=["user_id", "order_id"], how="left")
    is_test = marked["is_test"].fillna(False).astype(bool)
    return marked.loc[~is_test, data.columns].copy(), marked.loc[is_test, data.columns].copy()


def fit_model(user_item: pd.DataFrame, user_info: pd.DataFrame, item_info: pd.DataFrame) -> CMF_implicit:
    model = CMF_implicit(
        k=48,
        k_user=12,
        k_item=12,
        lambda_=3.0,
        alpha=8.0,
        w_main=1.0,
        w_user=3.0,
        w_item=3.0,
        niter=15,
        max_cg_steps=5,
        random_state=RANDOM_STATE,
        verbose=True,
        produce_dicts=True,
        nthreads=-1,
    )
    model.fit(
        user_item[["UserId", "ItemId", "Value"]],
        U=user_info,
        I=item_info,
    )
    return model


def recommend(model: CMF_implicit, user_item: pd.DataFrame, item_lookup: pd.DataFrame, users: list[int], n: int = TOP_N) -> pd.DataFrame:
    seen_items = user_item.groupby("UserId")["ItemId"].apply(lambda x: set(x.astype(int)))
    lookup = item_lookup.set_index("ItemId")
    rows = []
    for user_id in users:
        exclude = list(seen_items.get(user_id, set()))
        try:
            items, scores = model.topN(user_id, n=n, exclude=exclude, output_score=True)
        except Exception:
            continue
        for rank, (item_id, score) in enumerate(zip(items, scores), start=1):
            meta = lookup.loc[item_id] if item_id in lookup.index else None
            rows.append(
                {
                    "user_id": int(user_id),
                    "rank": rank,
                    "recommended_product_id": int(item_id),
                    "score": float(score),
                    "product_name": None if meta is None else meta["product_name"],
                    "category": None if meta is None else meta["category"],
                    "brand": None if meta is None else meta["brand"],
                    "item_quality_risk": None if meta is None else float(meta["item_quality_risk"]),
                }
            )
    return pd.DataFrame(rows)


def rank_metrics(recommendations: pd.DataFrame, test: pd.DataFrame, k: int = TOP_N) -> dict:
    true_items = test.groupby("user_id")["product_id"].apply(lambda x: set(x.astype(int)))
    true_categories = test.groupby("user_id")["category"].apply(lambda x: set(x.dropna().astype(str)))

    product_hits = []
    product_precision = []
    product_recall = []
    product_mrr = []
    category_hits = []
    category_precision = []
    category_recall = []
    category_mrr = []

    for user_id, group in recommendations.groupby("user_id"):
        if user_id not in true_items:
            continue
        group = group.sort_values("rank").head(k)
        rec_items = group["recommended_product_id"].astype(int).tolist()
        rec_categories = group["category"].dropna().astype(str).tolist()
        item_truth = true_items[user_id]
        category_truth = true_categories[user_id]

        item_hit_positions = [idx + 1 for idx, item in enumerate(rec_items) if item in item_truth]
        category_hit_positions = [idx + 1 for idx, category in enumerate(rec_categories) if category in category_truth]

        product_hits.append(float(bool(item_hit_positions)))
        product_precision.append(len(set(rec_items) & item_truth) / k)
        product_recall.append(len(set(rec_items) & item_truth) / len(item_truth))
        product_mrr.append(0.0 if not item_hit_positions else 1 / item_hit_positions[0])

        category_hits.append(float(bool(category_hit_positions)))
        category_precision.append(len(set(rec_categories) & category_truth) / k)
        category_recall.append(len(set(rec_categories) & category_truth) / len(category_truth))
        category_mrr.append(0.0 if not category_hit_positions else 1 / category_hit_positions[0])

    return {
        "evaluated_users": int(len(product_hits)),
        "k": k,
        "product_hit_rate_at_k": float(np.mean(product_hits)) if product_hits else 0.0,
        "product_precision_at_k": float(np.mean(product_precision)) if product_precision else 0.0,
        "product_recall_at_k": float(np.mean(product_recall)) if product_recall else 0.0,
        "product_mrr_at_k": float(np.mean(product_mrr)) if product_mrr else 0.0,
        "category_hit_rate_at_k": float(np.mean(category_hits)) if category_hits else 0.0,
        "category_precision_at_k": float(np.mean(category_precision)) if category_precision else 0.0,
        "category_recall_at_k": float(np.mean(category_recall)) if category_recall else 0.0,
        "category_mrr_at_k": float(np.mean(category_mrr)) if category_mrr else 0.0,
    }


def select_recommendation_users(data: pd.DataFrame, user_item: pd.DataFrame, limit: int = RECOMMEND_USERS) -> list[int]:
    latest = data["created_at"].max()
    user_stats = data.groupby("user_id").agg(
        recency_days=("created_at", lambda x: (latest - x.max()).days),
        orders=("order_id", "nunique"),
        spend=("sale_price", "sum"),
        cancel_rate=("is_cancelled", "mean"),
        return_rate=("is_returned", "mean"),
    )
    score = (
        minmax(user_stats["recency_days"]) * 0.45
        + minmax(user_stats["cancel_rate"]) * 0.25
        + minmax(user_stats["return_rate"]) * 0.20
        + (1 - minmax(user_stats["orders"])) * 0.10
    )
    known_users = set(user_item["UserId"].unique())
    users = score.sort_values(ascending=False).index
    return [int(user) for user in users if user in known_users][:limit]


def main() -> None:
    ensure_dirs()
    data = read_orders()

    train, test = temporal_split(data)
    user_info = build_user_info(train)
    item_info, item_lookup = build_item_info(train)
    user_item = build_user_item(train)

    user_info.to_csv(ARTIFACTS_DIR / "user_info.csv", index=False)
    item_info.to_csv(ARTIFACTS_DIR / "item_info.csv", index=False)
    item_lookup.to_csv(ARTIFACTS_DIR / "item_lookup.csv", index=False)
    user_item.to_csv(ARTIFACTS_DIR / "user_item.csv", index=False)

    model = fit_model(user_item, user_info, item_info)
    with (ARTIFACTS_DIR / "cmf_implicit_model.pkl").open("wb") as file:
        pickle.dump(model, file)

    eval_users = test["user_id"].drop_duplicates().astype(int).head(MAX_EVAL_USERS).tolist()
    eval_recommendations = recommend(model, user_item, item_lookup, eval_users, n=TOP_N)
    eval_metrics = rank_metrics(eval_recommendations, test, k=TOP_N)
    eval_recommendations.to_csv(ARTIFACTS_DIR / "hybrid_als_eval_recommendations.csv", index=False)

    recommendation_users = select_recommendation_users(train, user_item, RECOMMEND_USERS)
    production_recommendations = recommend(model, user_item, item_lookup, recommendation_users, n=TOP_N)
    production_recommendations.to_csv(ARTIFACTS_DIR / "hybrid_als_recommendations.csv", index=False)

    metrics = {
        "model": "cmfrec.CMF_implicit",
        "model_type": "hybrid implicit ALS / collective matrix factorization",
        "top_n": TOP_N,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "user_info_shape": list(user_info.shape),
        "item_info_shape": list(item_info.shape),
        "user_item_shape": list(user_item.shape),
        "production_recommendation_rows": int(len(production_recommendations)),
        "production_recommended_users": int(production_recommendations["user_id"].nunique()),
        "eval": eval_metrics,
    }
    (ARTIFACTS_DIR / "hybrid_als_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
