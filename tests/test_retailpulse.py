
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
SEGMENTS_FILE = ROOT / "reports" / "customer_segments.csv"
SEGMENT_SUMMARY_FILE = ROOT / "reports" / "segment_summary.csv"
FORECAST_FILE = ROOT / "reports" / "forecast_test_results.csv"
FORECAST_METRICS_FILE = ROOT / "reports" / "forecast_metrics.csv"
INVENTORY_FILE = ROOT / "reports" / "inventory_recommendations.csv"


def test_cleaned_dataset_exists():
    assert DATA_FILE.exists()


def test_cleaned_dataset_has_required_columns():
    df = pd.read_csv(DATA_FILE, nrows=5)
    required = {
        "InvoiceNo",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
        "Revenue",
    }
    assert required.issubset(df.columns)


def test_cleaned_sales_are_valid():
    df = pd.read_csv(DATA_FILE)
    assert not df.empty
    assert (df["Quantity"] > 0).all()
    assert (df["UnitPrice"] > 0).all()
    assert (df["Revenue"] > 0).all()


def test_customer_segments_cover_customers():
    assert SEGMENTS_FILE.exists()
    segments = pd.read_csv(SEGMENTS_FILE)
    assert {"CustomerID", "Segment"}.issubset(segments.columns)
    assert segments["CustomerID"].nunique() > 0
    assert segments["Segment"].notna().all()


def test_segment_summary_exists():
    assert SEGMENT_SUMMARY_FILE.exists()
    summary = pd.read_csv(SEGMENT_SUMMARY_FILE)
    assert not summary.empty


def test_forecast_results_are_valid():
    assert FORECAST_FILE.exists()
    forecast = pd.read_csv(FORECAST_FILE)

    required = {
        "date",
        "actual_revenue",
        "predicted_revenue",
        "baseline_lag7_revenue",
    }
    assert required.issubset(forecast.columns)
    assert len(forecast) > 0
    assert forecast["predicted_revenue"].notna().all()


def test_forecast_metrics_exist():
    assert FORECAST_METRICS_FILE.exists()
    metrics = pd.read_csv(FORECAST_METRICS_FILE)
    assert {"metric", "value"}.issubset(metrics.columns)

    values = dict(zip(metrics["metric"], metrics["value"]))
    assert values["MAE"] >= 0
    assert values["RMSE"] >= 0


def test_inventory_recommendations_are_valid():
    assert INVENTORY_FILE.exists()
    inventory = pd.read_csv(INVENTORY_FILE)

    required = {
        "StockCode",
        "Description",
        "AvgDailyDemand",
        "ReorderPoint",
        "RecommendedOrderQty",
        "InventoryStatus",
    }
    assert required.issubset(inventory.columns)
    assert not inventory.empty
    assert (inventory["AvgDailyDemand"] >= 0).all()
    assert (inventory["ReorderPoint"] >= 0).all()
    assert (inventory["RecommendedOrderQty"] >= 0).all()
