from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


API_BASE = os.getenv("FINAL_LENDING_API_URL", "http://127.0.0.1:8000")
API_TIMEOUT = int(os.getenv("FINAL_LENDING_API_TIMEOUT", "600"))


def api_get(path: str) -> dict:
    response = requests.get(f"{API_BASE}{path}", timeout=API_TIMEOUT)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Customer Experience Demo", layout="wide")
st.title("Customer Experience Demo")
st.caption("Streamlit shell over FastAPI endpoints for churn, recommendations, and sales forecast.")

with st.sidebar:
    st.subheader("API")
    st.code(API_BASE)
    if st.button("Check health"):
        st.json(api_get("/health"))

status_col1, status_col2 = st.columns(2)
status_col1.metric("API base", API_BASE)

try:
    api_get("/health")
    status_col2.success("API is reachable")
except requests.exceptions.RequestException as exc:
    status_col2.error("API is unavailable")
    st.error(f"Cannot reach API.\n\nDetails: {exc}")
    st.stop()

summary = None
if st.button("Load platform summary"):
    try:
        with st.spinner("Loading summary and warming caches..."):
            summary = api_get("/summary")
    except requests.exceptions.RequestException as exc:
        st.error(f"Summary loading failed.\n\nDetails: {exc}")

if summary is not None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Users", summary["users"])
    col2.metric("Recommendations", summary["recommendation_rows"])
    col3.metric("Forecast months", summary["forecast_months"])
    st.dataframe(pd.DataFrame(summary["top_features"]), use_container_width=True)

try:
    with st.spinner("Loading user ids..."):
        users = api_get("/users?limit=200")["user_ids"]
except requests.exceptions.RequestException as exc:
    st.error(f"Could not load users.\n\nDetails: {exc}")
    st.stop()

user_id = st.selectbox("Select user_id", users)

left, right = st.columns([1, 1])

with left:
    st.subheader("Churn prediction")
    if st.button("Run churn inference"):
        churn = api_get(f"/predict_churn/{user_id}")
        st.metric("Churn probability", f"{churn['churn_probability']:.3f}")
        st.write("Risk group:", churn["risk_group"])
        if "customer_features" in churn:
            st.dataframe(pd.DataFrame([churn["customer_features"]]), use_container_width=True)

with right:
    st.subheader("Recommendations")
    if st.button("Load recommendations"):
        recs = api_get(f"/recommend/{user_id}?top_n=5")["recommendations"]
        st.dataframe(pd.DataFrame(recs), use_container_width=True)

st.subheader("Sales forecast")
if st.button("Load sales forecast"):
    forecast = pd.DataFrame(api_get("/forecast_sales")["forecast"])
    if not forecast.empty:
        forecast["month"] = pd.to_datetime(forecast["month"])
        st.line_chart(forecast.set_index("month")["forecast_revenue"])
        st.dataframe(forecast, use_container_width=True)
    else:
        st.info("No forecast data available.")
