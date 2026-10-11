
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
SEGMENTS_FILE = ROOT / "reports" / "customer_segments.csv"
SEGMENT_SUMMARY_FILE = ROOT / "reports" / "segment_summary.csv"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RetailPulse | Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 RetailPulse")
st.caption("AI-powered retail analytics and customer intelligence")


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

if not DATA_FILE.exists():
    st.error("Cleaned dataset not found. Run the ETL pipeline first.")
    st.stop()

if not SEGMENTS_FILE.exists():
    st.error("Customer segments not found. Run customer segmentation first.")
    st.stop()

if not SEGMENT_SUMMARY_FILE.exists():
    st.error("Segment summary not found. Run customer segmentation first.")
    st.stop()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    transactions = pd.read_csv(
        DATA_FILE,
        parse_dates=["InvoiceDate"],
    )

    segments = pd.read_csv(SEGMENTS_FILE)
    summary = pd.read_csv(SEGMENT_SUMMARY_FILE, index_col=0)

    transactions["CustomerID"] = pd.to_numeric(
        transactions["CustomerID"], errors="coerce"
    )

    segments["CustomerID"] = pd.to_numeric(
        segments["CustomerID"], errors="coerce"
    )

    transactions = transactions.merge(
        segments[["CustomerID", "Segment"]],
        on="CustomerID",
        how="left",
    )

    return transactions, segments, summary


df, segments, segment_summary = load_data()


# --------------------------------------------------
# HTML TABLE HELPER — NO PYARROW REQUIRED
# --------------------------------------------------

def render_html_table(dataframe, max_rows=None):
    """Render a pandas DataFrame as an HTML table in Streamlit."""
    display_df = dataframe.copy()

    if max_rows is not None:
        display_df = display_df.head(max_rows)

    # Format dates and numeric values for display.
    for column in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[column]):
            display_df[column] = display_df[column].dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        elif pd.api.types.is_float_dtype(display_df[column]):
            display_df[column] = display_df[column].map(
                lambda value: (
                    f"{value:,.2f}" if pd.notna(value) else ""
                )
            )

    html = display_df.to_html(
        index=False,
        border=0,
        classes="retailpulse-table",
        escape=True,
    )

    st.markdown(
        """
        <style>
        .retailpulse-table-container {
            width: 100%;
            overflow-x: auto;
            border: 1px solid #3a3b44;
            border-radius: 8px;
        }

        .retailpulse-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .retailpulse-table th {
            background-color: #262730;
            color: #ffffff;
            padding: 11px;
            text-align: left;
            white-space: nowrap;
            border-bottom: 1px solid #454650;
        }

        .retailpulse-table td {
            padding: 10px;
            border-bottom: 1px solid #353640;
            text-align: left;
            white-space: nowrap;
        }

        .retailpulse-table tr:hover td {
            background-color: rgba(128, 128, 128, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="retailpulse-table-container">{html}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

countries = sorted(df["Country"].dropna().unique().tolist())

default_countries = (
    ["United Kingdom"]
    if "United Kingdom" in countries
    else countries[:5]
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=countries,
    default=default_countries,
)

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Transaction date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

filtered = df.copy()

if selected_countries:
    filtered = filtered[
        filtered["Country"].isin(selected_countries)
    ]
else:
    filtered = filtered.iloc[0:0]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

    filtered = filtered[
        filtered["InvoiceDate"].dt.date.between(
            start_date,
            end_date,
        )
    ]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

revenue = filtered["Revenue"].sum()
orders = filtered["InvoiceNo"].nunique()
customers = filtered["CustomerID"].nunique()
average_order_value = revenue / orders if orders else 0

st.subheader("Business Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Revenue", f"£{revenue:,.2f}")
c2.metric("Total Orders", f"{orders:,}")
c3.metric("Unique Customers", f"{customers:,}")
c4.metric("Average Order Value", f"£{average_order_value:,.2f}")

st.divider()


# --------------------------------------------------
# MONTHLY REVENUE AND COUNTRY ANALYSIS
# --------------------------------------------------

st.subheader("Sales Performance")

left, right = st.columns(2)

with left:
    monthly = (
        filtered.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly,
        x="InvoiceDate",
        y="Revenue",
        markers=True,
        title="Monthly Revenue Trend",
        labels={
            "InvoiceDate": "Month",
            "Revenue": "Revenue (£)",
        },
    )

    st.plotly_chart(fig, width="stretch")

with right:
    country_revenue = (
        filtered.groupby("Country", as_index=False)["Revenue"]
        .sum()
        .nlargest(10, "Revenue")
        .sort_values("Revenue")
    )

    fig = px.bar(
        country_revenue,
        x="Revenue",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Revenue",
        labels={"Revenue": "Revenue (£)"},
    )

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# PRODUCT AND CUSTOMER SEGMENT ANALYSIS
# --------------------------------------------------

st.subheader("Product and Customer Intelligence")

left, right = st.columns(2)

with left:
    product_revenue = (
        filtered.groupby("Description", as_index=False)["Revenue"]
        .sum()
        .nlargest(10, "Revenue")
        .sort_values("Revenue")
    )

    fig = px.bar(
        product_revenue,
        x="Revenue",
        y="Description",
        orientation="h",
        title="Top 10 Products by Revenue",
        labels={"Revenue": "Revenue (£)"},
    )

    st.plotly_chart(fig, width="stretch")

with right:
    segment_data = filtered.dropna(subset=["Segment"]).copy()

    segment_data["Segment"] = segment_data["Segment"].astype(int)

    segment_revenue = (
        segment_data.groupby("Segment", as_index=False)
        .agg(
            Revenue=("Revenue", "sum"),
            Customers=("CustomerID", "nunique"),
        )
    )

    fig = px.bar(
        segment_revenue,
        x="Segment",
        y="Revenue",
        title="Revenue by Customer Segment",
        hover_data=["Customers"],
        labels={
            "Revenue": "Revenue (£)",
            "Segment": "Cluster ID",
        },
    )

    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------
# CUSTOMER SEGMENT PROFILE
# --------------------------------------------------

st.subheader("Customer Segment Profile")

st.caption(
    "Cluster IDs are model-generated labels. "
    "Use recency, frequency and monetary averages to interpret them."
)

render_html_table(segment_summary.reset_index())


# --------------------------------------------------
# TRANSACTION EXPLORER
# --------------------------------------------------

st.subheader("Transaction Explorer")

st.caption("Showing up to 100 matching transactions.")

transaction_columns = [
    "InvoiceNo",
    "InvoiceDate",
    "Description",
    "Quantity",
    "UnitPrice",
    "Revenue",
    "CustomerID",
    "Country",
    "Segment",
]

render_html_table(
    filtered[transaction_columns],
    max_rows=100,
)


# --------------------------------------------------
# DEMAND FORECASTING RESULTS
# --------------------------------------------------

st.divider()
st.header("📈 Demand Forecasting")

FORECAST_FILE = ROOT / "reports" / "forecast_test_results.csv"
METRICS_FILE = ROOT / "reports" / "forecast_metrics.csv"

if FORECAST_FILE.exists() and METRICS_FILE.exists():
    forecast_df = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
    metrics_df = pd.read_csv(METRICS_FILE)

    st.caption(
        "Historical holdout evaluation: actual revenue compared "
        "with model predictions. These are test-period results, "
        "not future forecasts."
    )

    metric_values = dict(
        zip(metrics_df["metric"], metrics_df["value"])
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast MAE",
        f"£{metric_values['MAE']:,.2f}",
    )
    col2.metric(
        "Forecast RMSE",
        f"£{metric_values['RMSE']:,.2f}",
    )
    col3.metric(
        "Nonzero-Day MAPE",
        f"{metric_values['MAPE_nonzero_days_percent']:.2f}%",
    )

    fig = px.line(
        forecast_df,
        x="date",
        y=["actual_revenue", "predicted_revenue"],
        title="Actual vs Predicted Daily Revenue",
        labels={
            "date": "Date",
            "value": "Revenue (£)",
            "variable": "Series",
        },
    )

    st.plotly_chart(fig, width="stretch")

    baseline_mae = metric_values["Baseline_lag7_MAE"]
    model_mae = metric_values["MAE"]

    if baseline_mae > 0:
        improvement = (
            (baseline_mae - model_mae) / baseline_mae
        ) * 100

        st.write(
            f"**7-day baseline MAE:** £{baseline_mae:,.2f}"
        )
        st.write(
            f"**MAE improvement vs baseline:** {improvement:.2f}%"
        )

    st.caption(
        "MAPE is calculated only on days with nonzero actual revenue. "
        "Forecast quality should be checked against additional baselines "
        "before production use."
    )

else:
    st.info(
        "Forecast results are not available yet. "
        "Run: python src/demand_forecasting.py"
    )

# --------------------------------------------------
# INVENTORY OPTIMIZATION
# --------------------------------------------------

st.divider()
st.header("📦 Inventory Optimization")

INVENTORY_FILE = ROOT / "reports" / "inventory_recommendations.csv"

if INVENTORY_FILE.exists():
    inventory_df = pd.read_csv(INVENTORY_FILE)

    st.caption(
        "Illustrative replenishment recommendations based on "
        "historical sales. Stock-on-hand is assumed to be zero "
        "and lead time is assumed to be seven days."
    )

    total_products = len(inventory_df)
    reorder_count = (
        inventory_df["InventoryStatus"] == "Reorder review"
    ).sum()

    col1, col2 = st.columns(2)
    col1.metric("Products Analyzed", f"{total_products:,}")
    col2.metric("Products Requiring Review", f"{reorder_count:,}")

    top_products = inventory_df.nlargest(
        10, "TotalUnitsSold"
    ).sort_values("TotalUnitsSold")

    fig = px.bar(
        top_products,
        x="TotalUnitsSold",
        y="Description",
        orientation="h",
        title="Top 10 Products by Historical Unit Sales",
        labels={"TotalUnitsSold": "Units Sold"},
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Replenishment Recommendations")

    display_columns = [
        "StockCode",
        "Description",
        "TotalUnitsSold",
        "AvgDailyDemand",
        "ReorderPoint",
        "RecommendedOrderQty",
        "InventoryStatus",
    ]

    # HTML table avoids PyArrow-dependent table widgets.
    inventory_html = inventory_df[display_columns].head(100).to_html(
        index=False,
        border=0,
        escape=True,
        classes="retailpulse-table",
    )

    st.markdown(
        '<div style="width:100%; overflow-x:auto;">'
        + inventory_html
        + "</div>",
        unsafe_allow_html=True,
    )

    inventory_csv = inventory_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Inventory Recommendations (CSV)",
        data=inventory_csv,
        file_name="retailpulse_inventory_recommendations.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Inventory results not found. Run "
        "python src/inventory_optimization.py first."
    )

# --------------------------------------------------
# DOWNLOAD FILTERED DATA
# --------------------------------------------------

st.subheader("Export Data")

csv_data = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Transactions (CSV)",
    data=csv_data,
    file_name="retailpulse_filtered_transactions.csv",
    mime="text/csv",
)

st.caption(
    "RetailPulse | Historical UCI Online Retail dataset. "
    "Historical patterns do not guarantee future outcomes."
)
