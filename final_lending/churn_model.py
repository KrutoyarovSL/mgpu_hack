from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


CHURN_GAP_DAYS = 120
FEATURE_WINDOW_DAYS = 180
SNAPSHOT_FREQ = "MS"


@dataclass
class ChurnBundle:
    latest_predictions: pd.DataFrame
    validation_metrics: dict[str, float | str | int]
    feature_importance: pd.DataFrame
    model_features: list[str]


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str)


def prepare_orders_for_churn(data: pd.DataFrame) -> pd.DataFrame:
    orders = data.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True, format="mixed")
    for col in ["returned_at", "shipped_at", "delivered_at"]:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce", utc=True, format="mixed")

    orders["sale_price"] = pd.to_numeric(orders["sale_price"], errors="coerce").fillna(0.0)
    orders["cost"] = pd.to_numeric(orders["cost"], errors="coerce").fillna(0.0)
    orders["num_of_item"] = pd.to_numeric(orders.get("num_of_item", 1), errors="coerce").fillna(1.0)
    orders["line_revenue"] = orders["sale_price"] * orders["num_of_item"]
    orders["line_profit"] = (orders["sale_price"] - orders["cost"]) * orders["num_of_item"]
    orders["margin_ratio"] = np.where(orders["sale_price"].eq(0), 0.0, (orders["sale_price"] - orders["cost"]) / orders["sale_price"])
    orders["is_returned"] = orders["returned_at"].notna().astype(int)
    orders["is_cancelled"] = orders["status"].eq("Cancelled").astype(int)
    orders["is_shipped"] = orders["shipped_at"].notna().astype(int)
    orders["is_delivered"] = orders["delivered_at"].notna().astype(int)
    orders["shipping_delay_days"] = (orders["shipped_at"] - orders["created_at"]).dt.total_seconds() / 86400
    orders["delivery_days"] = (orders["delivered_at"] - orders["created_at"]).dt.total_seconds() / 86400
    orders["return_days"] = (orders["returned_at"] - orders["created_at"]).dt.total_seconds() / 86400
    orders["customer_review"] = _clean_text(orders.get("customer_review", pd.Series(index=orders.index, dtype=object)))
    return orders.dropna(subset=["user_id", "created_at"]).copy()


def prepare_events_for_churn(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["created_at"] = pd.to_datetime(work["created_at"], errors="coerce", utc=True, format="mixed")
    work["user_id"] = pd.to_numeric(work["user_id"], errors="coerce")
    work = work.dropna(subset=["user_id", "created_at"]).copy()
    work["user_id"] = work["user_id"].astype(int)
    work["event_date"] = work["created_at"].dt.date
    return work


def build_snapshot_features(
    orders_df: pd.DataFrame,
    events_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    feature_window_days: int = FEATURE_WINDOW_DAYS,
    churn_gap_days: int = CHURN_GAP_DAYS,
) -> pd.DataFrame:
    window_start = as_of_date - pd.Timedelta(days=feature_window_days)
    future_end = as_of_date + pd.Timedelta(days=churn_gap_days)

    hist_orders = orders_df[(orders_df["created_at"] >= window_start) & (orders_df["created_at"] < as_of_date)].copy()
    future_orders = orders_df[(orders_df["created_at"] >= as_of_date) & (orders_df["created_at"] < future_end)].copy()

    if hist_orders.empty:
        return pd.DataFrame()

    customer_features = hist_orders.groupby("user_id", as_index=False).agg(
        first_order_in_window=("created_at", "min"),
        last_order_in_window=("created_at", "max"),
        orders_cnt=("order_id", "nunique"),
        items_cnt=("num_of_item", "sum"),
        total_revenue=("line_revenue", "sum"),
        total_profit=("line_profit", "sum"),
        avg_order_value=("line_revenue", "mean"),
        avg_items_per_order_line=("num_of_item", "mean"),
        avg_margin_ratio=("margin_ratio", "mean"),
        return_rate=("is_returned", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        shipped_rate=("is_shipped", "mean"),
        delivered_rate=("is_delivered", "mean"),
        avg_shipping_delay_days=("shipping_delay_days", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_return_days=("return_days", "mean"),
        unique_categories=("category", "nunique"),
        unique_products=("product_id", "nunique"),
        unique_brands=("brand", "nunique"),
        favorite_category=("category", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        favorite_brand=("brand", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        traffic_source=("traffic_source", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        gender=("gender", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        state=("state", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        age=("age", "median"),
        is_loyal=("is_loyal", "max"),
    )

    customer_features["recency_days"] = (as_of_date - customer_features["last_order_in_window"]).dt.days
    customer_features["tenure_days_in_window"] = (
        customer_features["last_order_in_window"] - customer_features["first_order_in_window"]
    ).dt.days.clip(lower=0)
    customer_features["purchase_frequency_30d"] = customer_features["orders_cnt"] / max(feature_window_days / 30, 1)

    for window, suffix in [(30, "30d"), (90, "90d")]:
        recent = hist_orders[hist_orders["created_at"] >= as_of_date - pd.Timedelta(days=window)]
        recent_feats = recent.groupby("user_id", as_index=False).agg(
            **{
                f"orders_cnt_{suffix}": ("order_id", "nunique"),
                f"revenue_{suffix}": ("line_revenue", "sum"),
                f"return_rate_{suffix}": ("is_returned", "mean"),
            }
        )
        customer_features = customer_features.merge(recent_feats, on="user_id", how="left")

    positive_words = {
        "good", "great", "excellent", "amazing", "perfect", "love", "loved", "happy",
        "comfortable", "durable", "nice", "solid", "recommend", "satisfied", "works",
    }
    negative_words = {
        "bad", "poor", "terrible", "awful", "disappointed", "broken", "late", "return",
        "cheap", "worst", "hate", "problem", "issue", "damaged", "small", "big", "fit",
    }

    hist_orders["review_text"] = _clean_text(hist_orders["customer_review"])
    hist_orders["review_text_clean"] = (
        hist_orders["review_text"]
        .str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    hist_orders["review_word_count"] = hist_orders["review_text_clean"].str.split().str.len()
    hist_orders["positive_word_count"] = hist_orders["review_text_clean"].apply(
        lambda x: sum(token in positive_words for token in x.split())
    )
    hist_orders["negative_word_count"] = hist_orders["review_text_clean"].apply(
        lambda x: sum(token in negative_words for token in x.split())
    )
    hist_orders["review_sentiment_score"] = hist_orders["positive_word_count"] - hist_orders["negative_word_count"]
    hist_orders["review_negative_flag"] = (hist_orders["negative_word_count"] > hist_orders["positive_word_count"]).astype(int)
    hist_orders["review_positive_flag"] = (hist_orders["positive_word_count"] > hist_orders["negative_word_count"]).astype(int)

    review_features = hist_orders.groupby("user_id", as_index=False).agg(
        user_review_count=("review_text", "count"),
        user_avg_review_len=("review_word_count", "mean"),
        user_negative_review_rate=("review_negative_flag", "mean"),
        user_positive_review_rate=("review_positive_flag", "mean"),
        user_avg_sentiment_score=("review_sentiment_score", "mean"),
    )
    customer_features = customer_features.merge(review_features, on="user_id", how="left")

    hist_events = events_df[(events_df["created_at"] >= window_start) & (events_df["created_at"] < as_of_date)].copy()
    if not hist_events.empty:
        event_features = hist_events.groupby("user_id", as_index=False).agg(
            events_cnt=("event_type", "size"),
            sessions_cnt=("session_id", "nunique"),
            active_days=("event_date", "nunique"),
            last_event_at=("created_at", "max"),
            favorite_event_type=("event_type", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
        )
        event_pivot = pd.crosstab(hist_events["user_id"], hist_events["event_type"])
        event_pivot.columns = [f"event_{str(col).lower()}_cnt" for col in event_pivot.columns]
        event_features = event_features.merge(event_pivot.reset_index(), on="user_id", how="left")
        event_features["days_since_last_event"] = (as_of_date - event_features["last_event_at"]).dt.days
        customer_features = customer_features.merge(event_features, on="user_id", how="left")

    future_activity = future_orders.groupby("user_id", as_index=False).agg(future_orders_cnt=("order_id", "nunique"))
    customer_features = customer_features.merge(future_activity, on="user_id", how="left")
    customer_features["future_orders_cnt"] = customer_features["future_orders_cnt"].fillna(0)
    customer_features["churn_target"] = (customer_features["future_orders_cnt"] == 0).astype(int)
    customer_features["snapshot_date"] = as_of_date
    return customer_features


def train_notebook_churn_model(orders: pd.DataFrame, events: pd.DataFrame) -> ChurnBundle:
    max_order_date = orders["created_at"].max().normalize()
    min_snapshot_date = orders["created_at"].min().normalize() + pd.Timedelta(days=FEATURE_WINDOW_DAYS)
    max_snapshot_date = max_order_date - pd.Timedelta(days=CHURN_GAP_DAYS)
    snapshot_dates = pd.date_range(min_snapshot_date, max_snapshot_date, freq=SNAPSHOT_FREQ)[-12:]

    frames = []
    for as_of_date in snapshot_dates:
        frame = build_snapshot_features(orders, events, as_of_date)
        if not frame.empty:
            frames.append(frame)
    churn_dataset = pd.concat(frames, ignore_index=True)
    churn_dataset = churn_dataset[churn_dataset["orders_cnt"] >= 2].copy()

    snapshot_cutoff = snapshot_dates[-3]
    train_df = churn_dataset[churn_dataset["snapshot_date"] < snapshot_cutoff].copy()
    valid_df = churn_dataset[churn_dataset["snapshot_date"] >= snapshot_cutoff].copy()

    drop_cols = [
        "user_id",
        "first_order_in_window",
        "last_order_in_window",
        "last_event_at",
        "future_orders_cnt",
        "snapshot_date",
        "churn_target",
    ]
    feature_cols = [col for col in churn_dataset.columns if col not in drop_cols]
    cat_cols = [col for col in feature_cols if train_df[col].dtype == "object"]
    num_cols = [col for col in feature_cols if col not in cat_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df["churn_target"].copy()
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df["churn_target"].copy()

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Unknown").astype(str)
        X_valid[col] = X_valid[col].fillna("Unknown").astype(str)
    for col in num_cols:
        fill_value = X_train[col].median()
        X_train[col] = X_train[col].fillna(fill_value)
        X_valid[col] = X_valid[col].fillna(fill_value)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        eval_metric="AUC",
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_cols)
    valid_pred = model.predict_proba(X_valid)[:, 1]

    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)

    latest_snapshot = churn_dataset["snapshot_date"].max()
    latest_df = churn_dataset[churn_dataset["snapshot_date"] == latest_snapshot].copy()
    X_latest = latest_df[feature_cols].copy()
    for col in cat_cols:
        X_latest[col] = X_latest[col].fillna("Unknown").astype(str)
    for col in num_cols:
        fill_value = X_train[col].median()
        X_latest[col] = X_latest[col].fillna(fill_value)
    latest_prob = model.predict_proba(X_latest)[:, 1]

    latest_predictions = latest_df[["user_id", "snapshot_date", "churn_target"]].copy()
    latest_predictions["churn_probability"] = latest_prob
    latest_predictions["risk_group"] = pd.cut(
        latest_prob,
        bins=[-0.01, 0.35, 0.65, 1.01],
        labels=["low", "medium", "high"],
    ).astype(str)

    validation_metrics = {
        "model": "CatBoostClassifier",
        "rows": int(len(churn_dataset)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "roc_auc": float(roc_auc_score(y_valid, valid_pred)),
        "pr_auc": float(average_precision_score(y_valid, valid_pred)),
        "latest_snapshot": str(latest_snapshot.date()),
    }

    return ChurnBundle(
        latest_predictions=latest_predictions.reset_index(drop=True),
        validation_metrics=validation_metrics,
        feature_importance=feature_importance.reset_index(drop=True),
        model_features=feature_cols,
    )
