
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "Online Retail.xlsx"
OUTPUT_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"


def run_etl():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    print("Loading raw dataset...")
    df = pd.read_excel(RAW_FILE)

    print(f"Original rows: {len(df):,}")
    print(f"Original columns: {list(df.columns)}")

    # Standardize column names.
    df.columns = df.columns.str.strip()

    # Remove duplicate transactions.
    df = df.drop_duplicates()

    # Remove rows with missing essential fields.
    required = [
        "InvoiceNo", "StockCode", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID"
    ]
    df = df.dropna(subset=required)

    # Remove cancelled invoices and invalid sales.
    invoice = df["InvoiceNo"].astype(str)
    df = df[~invoice.str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Prepare dates and revenue.
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype("int64")
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Save cleaned data.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    # Print validation metrics.
    print("\nETL completed successfully!")
    print(f"Cleaned rows: {len(df):,}")
    print(f"Unique customers: {df['CustomerID'].nunique():,}")
    print(f"Unique products: {df['StockCode'].nunique():,}")
    print(f"Total revenue: {df['Revenue'].sum():,.2f}")
    print(f"Date range: {df['InvoiceDate'].min()} to "
          f"{df['InvoiceDate'].max()}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_etl()
