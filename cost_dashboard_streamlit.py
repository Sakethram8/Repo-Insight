#!/usr/bin/env python3
"""
Interactive Cost Dashboard using Streamlit
Real-time visualization of Gemini API usage and costs

Run with: streamlit run cost_dashboard_streamlit.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from cost_dashboard import CostTracker, GEMINI_PRICING

# Page config
st.set_page_config(
    page_title="Repo-Insight Cost Dashboard",
    page_icon="💰",
    layout="wide"
)

# Initialize tracker
@st.cache_resource
def get_tracker():
    return CostTracker()

tracker = get_tracker()

# Title and header
st.title("💰 Gemini API Cost Dashboard")
st.markdown("Real-time tracking of Google Gemini API usage for Repo-Insight")

# Refresh button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🔄 Refresh Data"):
        tracker.load()
        st.rerun()
with col2:
    if st.button("🗑️ Reset Data"):
        if st.session_state.get('confirm_reset'):
            tracker.reset()
            st.success("Data reset successfully!")
            st.session_state.confirm_reset = False
            st.rerun()
        else:
            st.session_state.confirm_reset = True
            st.warning("Click again to confirm reset")

# Get summary
summary = tracker.get_summary()

# Top metrics
st.markdown("---")
st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total API Calls",
        value=f"{summary.total_calls:,}",
        delta=None
    )

with col2:
    st.metric(
        label="Total Cost",
        value=f"${summary.total_cost:.4f}",
        delta=None
    )

with col3:
    st.metric(
        label="Total Tokens",
        value=f"{summary.total_input_tokens + summary.total_output_tokens:,}",
        delta=None
    )

with col4:
    st.metric(
        label="Avg Cost/Call",
        value=f"${summary.average_cost_per_call:.6f}",
        delta=None
    )

# Cost breakdown charts
st.markdown("---")
st.subheader("💸 Cost Breakdown")

col1, col2 = st.columns(2)

with col1:
    # Cost by operation
    if summary.cost_by_operation:
        df_ops = pd.DataFrame([
            {"Operation": op, "Cost": cost}
            for op, cost in summary.cost_by_operation.items()
        ])
        fig_ops = px.pie(
            df_ops,
            values="Cost",
            names="Operation",
            title="Cost by Operation",
            hole=0.4
        )
        fig_ops.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_ops, use_container_width=True)
    else:
        st.info("No operation data available yet")

with col2:
    # Cost by model
    if summary.cost_by_model:
        df_models = pd.DataFrame([
            {"Model": GEMINI_PRICING.get(model, {}).get("name", model), "Cost": cost}
            for model, cost in summary.cost_by_model.items()
        ])
        fig_models = px.pie(
            df_models,
            values="Cost",
            names="Model",
            title="Cost by Model",
            hole=0.4
        )
        fig_models.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_models, use_container_width=True)
    else:
        st.info("No model data available yet")

# Token usage
st.markdown("---")
st.subheader("🔢 Token Usage")

col1, col2 = st.columns(2)

with col1:
    # Token breakdown
    token_data = {
        "Type": ["Input Tokens", "Output Tokens"],
        "Count": [summary.total_input_tokens, summary.total_output_tokens]
    }
    df_tokens = pd.DataFrame(token_data)
    fig_tokens = px.bar(
        df_tokens,
        x="Type",
        y="Count",
        title="Input vs Output Tokens",
        color="Type",
        color_discrete_map={"Input Tokens": "#636EFA", "Output Tokens": "#EF553B"}
    )
    st.plotly_chart(fig_tokens, use_container_width=True)

with col2:
    # Tokens by operation
    if summary.tokens_by_operation:
        df_op_tokens = pd.DataFrame([
            {"Operation": op, "Tokens": tokens}
            for op, tokens in summary.tokens_by_operation.items()
        ])
        fig_op_tokens = px.bar(
            df_op_tokens,
            x="Operation",
            y="Tokens",
            title="Tokens by Operation",
            color="Tokens",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_op_tokens, use_container_width=True)
    else:
        st.info("No token data by operation available yet")

# Hourly breakdown
st.markdown("---")
st.subheader("📈 Hourly Cost Trend (Last 24 Hours)")

hourly_data = tracker.get_hourly_breakdown(hours=24)
if hourly_data:
    df_hourly = pd.DataFrame([
        {"Hour": hour, "Cost": cost}
        for hour, cost in hourly_data.items()
    ])
    fig_hourly = px.line(
        df_hourly,
        x="Hour",
        y="Cost",
        title="Cost Over Time",
        markers=True
    )
    fig_hourly.update_layout(
        xaxis_title="Time",
        yaxis_title="Cost ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
else:
    st.info("No hourly data available yet. Start using Gemini API to see trends.")

# Top expensive calls
st.markdown("---")
st.subheader("💎 Most Expensive API Calls")

top_calls = tracker.get_top_expensive_calls(limit=10)
if top_calls:
    df_expensive = pd.DataFrame([
        {
            "Timestamp": entry.timestamp[:19],  # Remove microseconds
            "Operation": entry.operation,
            "Model": GEMINI_PRICING.get(entry.model, {}).get("name", entry.model),
            "Input Tokens": entry.input_tokens,
            "Output Tokens": entry.output_tokens,
            "Cost": f"${entry.total_cost:.6f}"
        }
        for entry in top_calls
    ])
    st.dataframe(df_expensive, use_container_width=True)
else:
    st.info("No API calls recorded yet")

# Recommendations
st.markdown("---")
st.subheader("💡 Cost Optimization Recommendations")

if summary.recommendations:
    for rec in summary.recommendations:
        if "🚨" in rec or "⚠️" in rec:
            st.warning(rec)
        elif "💡" in rec:
            st.info(rec)
        else:
            st.success(rec)
else:
    st.success("✅ No recommendations at this time. Cost usage is optimal!")

# Pricing reference
st.markdown("---")
st.subheader("💵 Gemini API Pricing Reference")

pricing_data = []
for model_id, pricing in GEMINI_PRICING.items():
    pricing_data.append({
        "Model": pricing["name"],
        "Input (per 1M tokens)": f"${pricing['input_per_1k'] * 1000:.2f}",
        "Output (per 1M tokens)": f"${pricing['output_per_1k'] * 1000:.2f}"
    })

df_pricing = pd.DataFrame(pricing_data)
st.table(df_pricing)

# Export options
st.markdown("---")
st.subheader("📥 Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Export to CSV"):
        tracker.export_csv("cost_report.csv")
        st.success("Exported to cost_report.csv")

with col2:
    if st.button("Save Current State"):
        tracker.save()
        st.success("Cost data saved successfully!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>Repo-Insight Cost Dashboard | TechEx Hackathon 2024</p>
    <p>Powered by Streamlit & Plotly</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Made with Bob
