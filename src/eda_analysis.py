
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"


def run_eda():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["InvoiceDate"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Transaction-level business metrics.
    invoice_revenue = df.groupby("InvoiceNo")["Revenue"].sum()
    customer_revenue = df.groupby("CustomerID")["Revenue"].sum()

    monthly = (
        df.set_index("InvoiceDate")
        .resample("MS")["Revenue"]
        .sum()
    )

    daily = (
        df.set_index("InvoiceDate")
        .resample("D")["Revenue"]
        .sum()
    )

    top_products = (
        df.groupby("Description", dropna=True)["Revenue"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    top_countries = (
        df.groupby("Country")["Revenue"]
        .sum()
        .nlargest(10)
        .sort_values()
    )

    kpis = {
        "cleaned_transactions": len(df),
        "unique_invoices": int(df["InvoiceNo"].nunique()),
        "unique_customers": int(df["CustomerID"].nunique()),
        "unique_products": int(df["StockCode"].nunique()),
        "total_revenue": round(float(df["Revenue"].sum()), 2),
        "average_order_value": round(float(invoice_revenue.mean()), 2),
        "average_customer_revenue": round(
            float(customer_revenue.mean()), 2
        ),
        "date_start": str(df["InvoiceDate"].min()),
        "date_end": str(df["InvoiceDate"].max()),
    }

    pd.DataFrame(
        list(kpis.items()), columns=["metric", "value"]
    ).to_csv(REPORT_DIR / "kpi_summary.csv", index=False)

    # Monthly revenue.
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "monthly_revenue.png", dpi=150)
    plt.close(fig)

    # Daily revenue.
    fig, ax = plt.subplots(figsize=(12, 5))
    daily.plot(ax=ax)
    ax.set_title("Daily Revenue Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "daily_revenue.png", dpi=150)
    plt.close(fig)

    # Top products.
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Products by Revenue")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "top_products.png", dpi=150)
    plt.close(fig)

    # Top countries.
    fig, ax = plt.subplots(figsize=(10, 6))
    top_countries.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 Countries by Revenue")
    ax.set_xlabel("Revenue")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "top_countries.png", dpi=150)
    plt.close(fig)

    print("\nEDA completed successfully!")
    print("\nKPI SUMMARY")
    for key, value in kpis.items():
        print(f"{key}: {value}")

    print(f"\nKPI file: {REPORT_DIR / 'kpi_summary.csv'}")
    print(f"Charts saved in: {CHART_DIR}")


if __name__ == "__main__":
    run_eda()
