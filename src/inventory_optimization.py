
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
REPORT_DIR = ROOT / "reports"


def run_inventory_analysis():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["InvoiceDate"])

    # Aggregate sales into daily product-level quantities.
    daily_sales = (
        df.groupby([
            pd.Grouper(key="InvoiceDate", freq="D"),
            "StockCode",
            "Description",
        ])["Quantity"]
        .sum()
        .reset_index()
    )

    daily_sales = daily_sales.rename(
        columns={"InvoiceDate": "Date", "Quantity": "UnitsSold"}
    )

    products = []

    for (stock_code, description), group in daily_sales.groupby(
        ["StockCode", "Description"]
    ):
        group = group.sort_values("Date")

        # Include dates without recorded sales as zero demand.
        full_dates = pd.date_range(
            daily_sales["Date"].min(),
            daily_sales["Date"].max(),
            freq="D",
        )

        series = (
            group.set_index("Date")["UnitsSold"]
            .reindex(full_dates, fill_value=0)
        )

        # Exclude unusual returns/negative net daily quantities from
        # this simple positive-demand estimate.
        positive_demand = series.clip(lower=0)

        avg_daily_demand = positive_demand.mean()
        daily_std = positive_demand.std(ddof=0)

        # Explicit planning assumptions; dataset has no real stock
        # on hand or supplier lead-time fields.
        assumed_lead_time_days = 7
        assumed_review_period_days = 7
        safety_factor = 1.65

        safety_stock = safety_factor * daily_std * np.sqrt(
            assumed_lead_time_days
        )

        reorder_point = (
            avg_daily_demand * assumed_lead_time_days
            + safety_stock
        )

        target_stock = avg_daily_demand * (
            assumed_lead_time_days + assumed_review_period_days
        ) + safety_stock

        # Illustrative stock-on-hand assumption for demonstration.
        assumed_stock_on_hand = 0

        recommended_order = max(
            0,
            int(np.ceil(target_stock - assumed_stock_on_hand)),
        )

        products.append({
            "StockCode": stock_code,
            "Description": description,
            "TotalUnitsSold": int(positive_demand.sum()),
            "AvgDailyDemand": round(float(avg_daily_demand), 3),
            "DailyDemandStd": round(float(daily_std), 3),
            "AssumedLeadTimeDays": assumed_lead_time_days,
            "AssumedStockOnHand": assumed_stock_on_hand,
            "SafetyStock": int(np.ceil(safety_stock)),
            "ReorderPoint": int(np.ceil(reorder_point)),
            "TargetStock": int(np.ceil(target_stock)),
            "RecommendedOrderQty": recommended_order,
        })

    inventory = pd.DataFrame(products)

    inventory["InventoryStatus"] = np.where(
        inventory["AvgDailyDemand"] <= 0,
        "No recorded positive demand",
        np.where(
            inventory["AssumedStockOnHand"] <= inventory["ReorderPoint"],
            "Reorder review",
            "Stock above reorder point",
        ),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "inventory_recommendations.csv"
    inventory = inventory.sort_values(
        "TotalUnitsSold", ascending=False
    )
    inventory.to_csv(output_file, index=False)

    print("\nInventory analysis completed!")
    print(f"Products analyzed: {len(inventory):,}")
    print(f"Output: {output_file}")
    print("\nTop 10 products by historical unit sales:")
    print(
        inventory[
            [
                "StockCode",
                "Description",
                "TotalUnitsSold",
                "AvgDailyDemand",
                "ReorderPoint",
                "RecommendedOrderQty",
                "InventoryStatus",
            ]
        ].head(10).to_string(index=False)
    )

    print(
        "\nIMPORTANT: Stock-on-hand is assumed to be zero and "
        "lead time is assumed to be 7 days. These are planning "
        "illustrations, not actual inventory records."
    )


if __name__ == "__main__":
    run_inventory_analysis()
