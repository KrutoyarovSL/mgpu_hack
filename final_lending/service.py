from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analytics_solution as core
from final_lending.churn_model import ChurnBundle, prepare_events_for_churn, prepare_orders_for_churn, train_notebook_churn_model

try:
    import hybrid_als_recommender as als
except Exception:  # pragma: no cover - optional dependency path
    als = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "m_pred" / "archive" / "data.csv"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "m_pred" / "archive" / "events.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
CORE_ARTIFACTS_DIR = ARTIFACTS_DIR / "core"
ALS_ARTIFACTS_DIR = ARTIFACTS_DIR / "hybrid_als"
PREBUILT_ALS_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "hybrid_als"
FEATURE_SLICE_PATH = CORE_ARTIFACTS_DIR / "customer_feature_slice.csv"
PREDICTIONS_PATH = CORE_ARTIFACTS_DIR / "churn_predictions.csv"
COEFFICIENTS_PATH = CORE_ARTIFACTS_DIR / "churn_feature_importance.csv"
FORECAST_PATH = CORE_ARTIFACTS_DIR / "sales_forecast.csv"
METRICS_PATH = CORE_ARTIFACTS_DIR / "churn_metrics.json"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def configure_core_paths() -> None:
    core.DATA_PATH = _env_path("FINAL_LENDING_DATA_PATH", DEFAULT_DATA_PATH)
    core.EVENTS_PATH = _env_path("FINAL_LENDING_EVENTS_PATH", DEFAULT_EVENTS_PATH)
    core.ARTIFACTS_DIR = CORE_ARTIFACTS_DIR
    core.DASHBOARD_PATH = ARTIFACTS_DIR / "dashboard.html"
    CORE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    core.ensure_dirs()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ALS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_als_paths() -> None:
    if als is None:
        return
    als.DATA_PATH = _env_path("FINAL_LENDING_DATA_PATH", DEFAULT_DATA_PATH)
    als.ARTIFACTS_DIR = ALS_ARTIFACTS_DIR
    als.ensure_dirs()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _core_cache_exists() -> bool:
    return all(path.exists() for path in [FEATURE_SLICE_PATH, PREDICTIONS_PATH, COEFFICIENTS_PATH, FORECAST_PATH, METRICS_PATH])


def _load_core_cache() -> dict[str, Any]:
    features = pd.read_csv(FEATURE_SLICE_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    sales_forecast = pd.read_csv(FORECAST_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    recommendations = _load_prebuilt_recommendations()
    if recommendations is None:
        recommendations = pd.DataFrame()
    return {
        "data": None,
        "features": features,
        "predictions": predictions,
        "metrics": metrics,
        "coefficients": coefficients,
        "category_sales": None,
        "product_quality": None,
        "recommendations": recommendations,
        "sales_forecast": sales_forecast,
    }


def _write_core_cache(
    feature_slice: pd.DataFrame,
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
    sales_forecast: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    CORE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_slice.to_csv(FEATURE_SLICE_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    coefficients.to_csv(COEFFICIENTS_PATH, index=False)
    sales_forecast.to_csv(FORECAST_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_prebuilt_recommendations() -> pd.DataFrame | None:
    candidates = [
        ALS_ARTIFACTS_DIR / "hybrid_als_recommendations.csv",
        PREBUILT_ALS_ARTIFACTS_DIR / "hybrid_als_recommendations.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


def _fallback_recommendations(
    user_id: int,
    predictions: pd.DataFrame,
    product_quality: pd.DataFrame,
) -> pd.DataFrame:
    pred_row = predictions.loc[predictions["user_id"].eq(user_id)]
    churn_probability = float(pred_row["churn_probability"].iat[0]) if not pred_row.empty else 0.0
    popularity = product_quality.sort_values(
        ["quality_risk", "revenue", "avg_margin"],
        ascending=[True, False, False],
    ).head(5)
    if popularity.empty:
        return pd.DataFrame(columns=["user_id", "rank", "recommended_product_id", "product_name", "category", "brand", "score", "item_quality_risk"])
    rows = []
    for rank, row in enumerate(popularity.itertuples(index=False), start=1):
        rows.append(
            {
                "user_id": user_id,
                "rank": rank,
                "recommended_product_id": int(row.product_id),
                "product_name": row.product_name,
                "category": row.category,
                "brand": row.brand,
                "score": 1.0 - float(row.quality_risk) + churn_probability * 0.1,
                "item_quality_risk": float(row.quality_risk),
                "source": "business_rule_fallback",
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def _load_als_model_bundle() -> dict[str, Any] | None:
    if als is None:
        return None
    artifact_root = ALS_ARTIFACTS_DIR if (ALS_ARTIFACTS_DIR / "cmf_implicit_model.pkl").exists() else PREBUILT_ALS_ARTIFACTS_DIR
    model_path = artifact_root / "cmf_implicit_model.pkl"
    user_item_path = artifact_root / "user_item.csv"
    item_lookup_path = artifact_root / "item_lookup.csv"
    if not (model_path.exists() and user_item_path.exists() and item_lookup_path.exists()):
        return None
    with model_path.open("rb") as file:
        model = pickle.load(file)
    user_item = pd.read_csv(user_item_path)
    item_lookup = pd.read_csv(item_lookup_path)
    return {"model": model, "user_item": user_item, "item_lookup": item_lookup}


@lru_cache(maxsize=1)
def load_core_bundle() -> dict[str, Any]:
    configure_core_paths()
    if _core_cache_exists():
        return _load_core_cache()

    data = core.read_orders()
    raw_orders = pd.read_csv(_env_path("FINAL_LENDING_DATA_PATH", DEFAULT_DATA_PATH), low_memory=False)
    raw_events = pd.read_csv(_env_path("FINAL_LENDING_EVENTS_PATH", DEFAULT_EVENTS_PATH), low_memory=False)

    order_features, category_sales, product_quality = core.build_order_features(data)
    event_features = core.aggregate_events()
    features = core.merge_features(order_features, event_features)
    features["customer_segment"] = core.add_customer_segments(features)

    churn_bundle: ChurnBundle = train_notebook_churn_model(
        prepare_orders_for_churn(raw_orders),
        prepare_events_for_churn(raw_events),
    )
    predictions = churn_bundle.latest_predictions.copy()
    metrics = churn_bundle.validation_metrics
    coefficients = churn_bundle.feature_importance.copy()

    feature_slice_cols = [
        "user_id",
        "orders",
        "total_spend",
        "return_rate",
        "cancel_rate",
        "recency_days",
        "purchase_frequency",
        "customer_segment",
        "abc_segment",
        "xyz_segment",
    ]
    feature_slice = features.reset_index().rename(columns={"index": "user_id"})
    feature_slice = feature_slice[[col for col in feature_slice_cols if col in feature_slice.columns]]
    predictions = predictions.merge(feature_slice, on="user_id", how="left")

    recommendations = _load_prebuilt_recommendations()
    if recommendations is None or recommendations.empty:
        recommendations = core.build_recommendations(data, features, predictions[["user_id", "churn_probability", "risk_group"]], product_quality)
    sales_forecast = core.forecast_sales(data)
    _write_core_cache(feature_slice, predictions, coefficients, sales_forecast, metrics)
    return {
        "data": data,
        "features": feature_slice,
        "predictions": predictions,
        "metrics": metrics,
        "coefficients": coefficients,
        "category_sales": category_sales,
        "product_quality": product_quality,
        "recommendations": recommendations,
        "sales_forecast": sales_forecast,
    }


def get_available_user_ids(limit: int = 200) -> list[int]:
    bundle = load_core_bundle()
    return bundle["predictions"]["user_id"].sort_values().head(limit).astype(int).tolist()


def get_summary() -> dict[str, Any]:
    bundle = load_core_bundle()
    predictions = bundle["predictions"]
    return {
        "users": int(predictions["user_id"].nunique()),
        "high_risk_share": float(predictions["risk_group"].eq("high").mean()),
        "recommendation_rows": int(len(bundle["recommendations"])),
        "forecast_months": int(len(bundle["sales_forecast"])),
        "churn_model": bundle["metrics"]["model"],
        "churn_auc": float(bundle["metrics"]["roc_auc"]),
        "top_features": [
            {key: _to_jsonable(value) for key, value in row.items()}
            for row in bundle["coefficients"].head(5).to_dict(orient="records")
        ],
    }


def get_user_churn(user_id: int) -> dict[str, Any]:
    bundle = load_core_bundle()
    predictions = bundle["predictions"]
    features = bundle["features"]
    pred_row = predictions.loc[predictions["user_id"].eq(user_id)]
    if pred_row.empty:
        raise KeyError(f"user_id={user_id} not found")
    feature_row = features.loc[features["user_id"].eq(user_id)]
    response = {
        key: _to_jsonable(value)
        for key, value in pred_row.iloc[0].to_dict().items()
    }
    if not feature_row.empty:
        cols = [
            "orders",
            "total_spend",
            "return_rate",
            "cancel_rate",
            "recency_days",
            "purchase_frequency",
            "customer_segment",
            "abc_segment",
            "xyz_segment",
        ]
        response["customer_features"] = {
            col: _to_jsonable(feature_row.iloc[0][col])
            for col in cols
            if col in feature_row.columns
        }
    return response


def get_user_recommendations(user_id: int, top_n: int = 5) -> list[dict[str, Any]]:
    bundle = load_core_bundle()
    prebuilt = _load_prebuilt_recommendations()
    if prebuilt is not None:
        rows = prebuilt.loc[prebuilt["user_id"].eq(user_id)].copy()
        if not rows.empty:
            rows["source"] = "hybrid_als_prebuilt"
            return [
                {key: _to_jsonable(value) for key, value in row.items()}
                for row in rows.head(top_n).to_dict(orient="records")
            ]

    configure_als_paths()
    als_bundle = _load_als_model_bundle()
    if als_bundle is not None and als is not None:
        recs = als.recommend(
            als_bundle["model"],
            als_bundle["user_item"],
            als_bundle["item_lookup"],
            [int(user_id)],
            n=top_n,
        )
        if not recs.empty:
            recs["source"] = "hybrid_als"
            return [
                {key: _to_jsonable(value) for key, value in row.items()}
                for row in recs.to_dict(orient="records")
            ]

    fallback = bundle["recommendations"].loc[bundle["recommendations"]["user_id"].eq(user_id)].copy()
    if fallback.empty:
        if bundle["product_quality"] is None:
            return []
        fallback = _fallback_recommendations(user_id, bundle["predictions"], bundle["product_quality"])
    else:
        fallback = fallback.rename(columns={"quality_risk": "item_quality_risk"})
        fallback["score"] = 1.0 - fallback["item_quality_risk"]
        fallback["source"] = "retention_rules"
    return [
        {key: _to_jsonable(value) for key, value in row.items()}
        for row in fallback.head(top_n).to_dict(orient="records")
    ]


def get_sales_forecast() -> list[dict[str, Any]]:
    bundle = load_core_bundle()
    return [
        {key: _to_jsonable(value) for key, value in row.items()}
        for row in bundle["sales_forecast"].to_dict(orient="records")
    ]
