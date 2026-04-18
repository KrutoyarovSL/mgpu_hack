from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
RECOMMENDATIONS_PATH = BASE_DIR / "artifacts" / "recommendations.csv"
PREDICTIONS_PATH = BASE_DIR / "artifacts" / "churn_predictions.csv"
OUTPUT_PATH = BASE_DIR / "artifacts" / "recommendation_metrics.json"

K = 3


def safe_float(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def quality_table(data: pd.DataFrame) -> pd.DataFrame:
    work = data.copy()
    work["is_returned"] = work["returned_at"].notna().astype(int)
    work["is_cancelled"] = work["status"].eq("Cancelled").astype(int)
    work["margin"] = work["sale_price"] - work["cost"]
    quality = work.groupby(["product_id", "product_name", "category", "brand"]).agg(
        revenue=("sale_price", "sum"),
        items=("product_id", "size"),
        return_rate=("is_returned", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    quality["quality_risk"] = quality["return_rate"] * 0.55 + quality["cancel_rate"] * 0.45
    return quality


def recommend_for_users(train: pd.DataFrame, product_quality: pd.DataFrame, users: list[int]) -> pd.DataFrame:
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
    top_by_category = {
        category: group.head(5)
        for category, group in popularity.groupby("category", sort=False)
    }
    global_top = popularity.head(30)
    user_categories = train.groupby("user_id")["category"].apply(lambda x: set(x.dropna().astype(str)))

    rows = []
    for user_id in users:
        seen = user_categories.get(user_id, set())
        candidates = []
        for category in seen:
            if category in top_by_category:
                candidates.append(top_by_category[category])
        candidates.append(global_top[~global_top["category"].isin(seen)].head(10))
        if not candidates:
            continue
        candidate_df = pd.concat(candidates).drop_duplicates("product_id").head(K)
        for rank, row in enumerate(candidate_df.itertuples(index=False), start=1):
            rows.append(
                {
                    "user_id": int(user_id),
                    "rank": rank,
                    "recommended_product_id": int(row.product_id),
                    "category": row.category,
                    "quality_risk": float(row.quality_risk),
                }
            )
    return pd.DataFrame(rows)


def rank_metric(recommended: list[int], relevant: set[int]) -> tuple[int, float, float]:
    hits = [idx + 1 for idx, item in enumerate(recommended) if item in relevant]
    hit = int(bool(hits))
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    return hit, reciprocal_rank, recall


def evaluate_temporal_holdout(data: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    order_dates = data.groupby(["user_id", "order_id"])["created_at"].max().reset_index()
    order_dates["order_rank_desc"] = order_dates.groupby("user_id")["created_at"].rank(method="first", ascending=False)
    latest_orders = order_dates[order_dates["order_rank_desc"].eq(1)][["user_id", "order_id"]]

    order_counts = order_dates.groupby("user_id")["order_id"].nunique()
    eligible_users = set(order_counts[order_counts >= 2].index.astype(int))

    high_risk_users = (
        predictions.sort_values("churn_probability", ascending=False)
        .head(5000)["user_id"]
        .astype(int)
        .tolist()
    )
    eval_users = [user for user in high_risk_users if user in eligible_users]

    latest_key = set(map(tuple, latest_orders[latest_orders["user_id"].isin(eval_users)][["user_id", "order_id"]].to_numpy()))
    is_test = data[["user_id", "order_id"]].apply(lambda row: (row["user_id"], row["order_id"]) in latest_key, axis=1)
    train = data.loc[~is_test].copy()
    test = data.loc[is_test].copy()

    product_quality = quality_table(train)
    recs = recommend_for_users(train, product_quality, eval_users)

    test_products = test.groupby("user_id")["product_id"].apply(lambda x: set(x.astype(int)))
    test_categories = test.groupby("user_id")["category"].apply(lambda x: set(x.dropna().astype(str)))

    product_hits_at_1 = []
    product_hits_at_k = []
    product_mrr = []
    product_recall = []
    category_hits_at_1 = []
    category_hits_at_k = []
    category_mrr = []
    category_recall = []
    precision_at_k = []
    evaluated = 0

    for user_id, group in recs.groupby("user_id"):
        if user_id not in test_products:
            continue
        recommended_products = group.sort_values("rank")["recommended_product_id"].astype(int).tolist()
        recommended_categories = group.sort_values("rank")["category"].astype(str).tolist()
        true_products = test_products[user_id]
        true_categories = test_categories[user_id]

        product_hit, product_rr, product_rec = rank_metric(recommended_products[:K], true_products)
        category_hit, category_rr, category_rec = rank_metric(recommended_categories[:K], true_categories)
        product_hits_at_1.append(int(recommended_products[:1][0] in true_products) if recommended_products else 0)
        product_hits_at_k.append(product_hit)
        product_mrr.append(product_rr)
        product_recall.append(product_rec)
        category_hits_at_1.append(int(recommended_categories[:1][0] in true_categories) if recommended_categories else 0)
        category_hits_at_k.append(category_hit)
        category_mrr.append(category_rr)
        category_recall.append(category_rec)
        precision_at_k.append(len(set(recommended_products[:K]) & true_products) / K)
        evaluated += 1

    return {
        "method": "temporal_holdout_latest_order",
        "k": K,
        "eligible_multi_order_users": len(eligible_users),
        "target_high_risk_users": len(high_risk_users),
        "evaluated_users": evaluated,
        "product_hit_rate_at_1": safe_float(np.mean(product_hits_at_1)),
        "product_hit_rate_at_3": safe_float(np.mean(product_hits_at_k)),
        "product_precision_at_3": safe_float(np.mean(precision_at_k)),
        "product_recall_at_3": safe_float(np.mean(product_recall)),
        "product_mrr_at_3": safe_float(np.mean(product_mrr)),
        "category_hit_rate_at_1": safe_float(np.mean(category_hits_at_1)),
        "category_hit_rate_at_3": safe_float(np.mean(category_hits_at_k)),
        "category_recall_at_3": safe_float(np.mean(category_recall)),
        "category_mrr_at_3": safe_float(np.mean(category_mrr)),
    }


def evaluate_current_artifact(data: pd.DataFrame, recommendations: pd.DataFrame) -> dict:
    total_products = data["product_id"].nunique()
    total_categories = data["category"].nunique()
    rec_users = recommendations["user_id"].nunique()
    unique_rec_products = recommendations["recommended_product_id"].nunique()
    unique_rec_categories = recommendations["category"].nunique()
    per_user_counts = recommendations.groupby("user_id")["recommended_product_id"].size()
    category_diversity = recommendations.groupby("user_id")["category"].nunique()

    return {
        "recommended_rows": int(len(recommendations)),
        "recommended_users": int(rec_users),
        "avg_recommendations_per_user": safe_float(per_user_counts.mean()),
        "min_recommendations_per_user": int(per_user_counts.min()),
        "max_recommendations_per_user": int(per_user_counts.max()),
        "unique_recommended_products": int(unique_rec_products),
        "unique_recommended_categories": int(unique_rec_categories),
        "catalog_coverage_product": safe_float(unique_rec_products / total_products),
        "catalog_coverage_category": safe_float(unique_rec_categories / total_categories),
        "avg_unique_categories_per_user": safe_float(category_diversity.mean()),
        "avg_quality_risk": safe_float(recommendations["quality_risk"].mean()),
        "median_quality_risk": safe_float(recommendations["quality_risk"].median()),
        "zero_quality_risk_share": safe_float((recommendations["quality_risk"] == 0).mean()),
    }


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        usecols=[
            "user_id",
            "order_id",
            "created_at",
            "product_id",
            "product_name",
            "category",
            "brand",
            "status",
            "returned_at",
            "sale_price",
            "cost",
        ],
        low_memory=False,
    )
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True, format="mixed")
    data["returned_at"] = pd.to_datetime(data["returned_at"], errors="coerce", utc=True, format="mixed")
    data["sale_price"] = pd.to_numeric(data["sale_price"], errors="coerce").fillna(0.0)
    data["cost"] = pd.to_numeric(data["cost"], errors="coerce").fillna(0.0)

    recommendations = pd.read_csv(RECOMMENDATIONS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    metrics = {
        "artifact_metrics": evaluate_current_artifact(data, recommendations),
        "temporal_holdout_metrics": evaluate_temporal_holdout(data, predictions),
        "notes": [
            "Product-level metrics are expected to be low because the catalog is large and the baseline recommends only three items.",
            "Category-level metrics are more informative for this business-rule baseline because it recommends safe popular products within or near a user's prior categories.",
            "The temporal holdout uses the latest order as future ground truth and rebuilds recommendation candidates from earlier history.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
