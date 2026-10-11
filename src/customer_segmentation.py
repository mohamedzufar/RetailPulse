
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"


def run_segmentation():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["InvoiceDate"])

    # Define the reference date as one day after the last transaction.
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    # Create customer-level RFM features.
    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
    )

    rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
    rfm = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # Log-transform skewed values before scaling.
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features["Recency"] = features["Recency"].clip(lower=0)
    features = features.apply(lambda column: __import__("numpy").log1p(column))

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Fit customer segments.
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Segment"] = model.fit_predict(scaled_features)

    # Save model artifacts and customer-level results.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "rfm_kmeans.joblib")
    joblib.dump(scaler, MODEL_DIR / "rfm_scaler.joblib")
    rfm.to_csv(REPORT_DIR / "customer_segments.csv", index_label="CustomerID")

    # Summarize the business meaning of each cluster.
    summary = rfm.groupby("Segment").agg(
        Customers=("Recency", "size"),
        AvgRecencyDays=("Recency", "mean"),
        AvgOrders=("Frequency", "mean"),
        AvgRevenue=("Monetary", "mean"),
        TotalRevenue=("Monetary", "sum"),
    ).round(2)

    summary.to_csv(REPORT_DIR / "segment_summary.csv")

    # Plot segment customer counts.
    fig, ax = plt.subplots(figsize=(8, 5))
    summary["Customers"].plot(kind="bar", ax=ax)
    ax.set_title("Customer Count by RFM Segment")
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "customer_segments.png", dpi=150)
    plt.close(fig)

    print("\nCustomer segmentation completed!")
    print(f"Reference date: {reference_date.date()}")
    print(f"Customers segmented: {len(rfm):,}")
    print("\nSegment summary:")
    print(summary)
    print(f"\nSaved: {REPORT_DIR / 'customer_segments.csv'}")
    print(f"Saved: {REPORT_DIR / 'segment_summary.csv'}")
    print(f"Saved: {MODEL_DIR / 'rfm_kmeans.joblib'}")


if __name__ == "__main__":
    run_segmentation()
