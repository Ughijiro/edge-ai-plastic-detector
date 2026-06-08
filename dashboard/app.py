import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from cloud.dynamodb_reader import get_all_events

st.set_page_config(
    page_title="Plastic Detector Dashboard",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Edge-AI Plastic Detector Dashboard")

st.write(
    "This dashboard displays detection and action events stored in AWS DynamoDB."
)

try:
    events = get_all_events()

    if not events:
        st.warning("No events found in DynamoDB yet.")
        st.stop()

    df = pd.DataFrame(events)

    st.subheader("Latest Event")
    latest_event = events[0]
    st.json(latest_event)

    total_events = len(df)
    total_collect = len(df[df["action"] == "COLLECT"]) if "action" in df else 0
    total_alarm = len(df[df["action"] == "ALARM"]) if "action" in df else 0
    total_stop = len(df[df["action"] == "STOP"]) if "action" in df else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Events", total_events)
    col2.metric("COLLECT Events", total_collect)
    col3.metric("ALARM Events", total_alarm)
    col4.metric("STOP Events", total_stop)

    st.subheader("Actions Summary")

    if "action" in df:
        action_counts = df["action"].value_counts()
        st.bar_chart(action_counts)

    st.subheader("Detected Object Labels")

    if "selected_label" in df:
        label_counts = df["selected_label"].dropna().value_counts()

        if not label_counts.empty:
            st.bar_chart(label_counts)
        else:
            st.info("No selected labels found yet.")

    st.subheader("Average Confidence")

    if "selected_confidence" in df:
        confidence_values = pd.to_numeric(
            df["selected_confidence"],
            errors="coerce"
        ).dropna()

        if not confidence_values.empty:
            avg_confidence = confidence_values.mean()
            st.metric("Average Selected Confidence", f"{avg_confidence:.2f}")
        else:
            st.info("No confidence values available yet.")

    st.subheader("Events Table")
    st.dataframe(df, use_container_width=True)

except Exception as error:
    st.error("Could not load events from DynamoDB.")
    st.exception(error)