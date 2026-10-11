
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "retail_cleaned.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
CHART_DIR = REPORT_DIR / "charts"


def create_daily_series(df):
    daily = (
        df.set_index("InvoiceDate")["Revenue"]
        .resample("D")
        .sum()
        .rename("Revenue")
        .to_frame()
    )

    # Missing calendar dates mean zero recorded revenue.
    daily["Revenue"] = daily["Revenue"].fillna(0)
    return daily


def add_features(daily):
    data = daily.copy()

    # Calendar features
    data["day_of_week"] = data.index.dayofweek
    data["day_of_month"] = data.index.day
    data["month"] = data.index.month
    data["day_of_year"] = data.index.dayofyear

    # Historical revenue features
    for lag in [1, 7, 14, 28]:
        data[f"lag_{lag}"] = data["Revenue"].shift(lag)

    for window in [7, 14, 28]:
        data[f"rolling_mean_{window}"] = (
            data["Revenue"]
            .shift(1)
            .rolling(window=window)
            .mean()
        )

    return data.dropna()


def run_forecasting():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["InvoiceDate"])

    daily = create_daily_series(df)
    supervised = add_features(daily)

    if len(supervised) < 90:
        raise ValueError(
            "Not enough daily history for this forecasting setup."
        )

    feature_columns = [
        "day_of_week",
        "day_of_month",
        "month",
        "day_of_year",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
    ]

    # Chronological split: never randomly shuffle time-series data.
    split_index = int(len(supervised) * 0.8)

    train = supervised.iloc[:split_index]
    test = supervised.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["Revenue"]
    X_test = test[feature_columns]
    y_test = test["Revenue"]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    nonzero = y_test != 0
    if nonzero.any():
        mape = (
            np.abs(
                (y_test[nonzero] - predictions[nonzero])
                / y_test[nonzero]
            ).mean()
            * 100
        )
    else:
        mape = np.nan

    # Naive baseline: predict revenue from seven days earlier.
    baseline_predictions = test["lag_7"].to_numpy()
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame({
        "date": test.index,
        "actual_revenue": y_test.to_numpy(),
        "predicted_revenue": predictions,
        "baseline_lag7_revenue": baseline_predictions,
    })

    results.to_csv(
        REPORT_DIR / "forecast_test_results.csv",
        index=False,
    )

    metrics = pd.DataFrame([
        {"metric": "MAE", "value": mae},
        {"metric": "RMSE", "value": rmse},
        {"metric": "MAPE_nonzero_days_percent", "value": mape},
        {"metric": "Baseline_lag7_MAE", "value": baseline_mae},
        {"metric": "Training_days", "value": len(train)},
        {"metric": "Test_days", "value": len(test)},
    ])

    metrics.to_csv(
        REPORT_DIR / "forecast_metrics.csv",
        index=False,
    )

    joblib.dump(model, MODEL_DIR / "demand_forecast_model.joblib")
    joblib.dump(feature_columns, MODEL_DIR / "forecast_features.joblib")

    # Actual vs predicted plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, y_test, label="Actual", linewidth=1.5)
    ax.plot(test.index, predictions, label="Random Forest forecast", linewidth=1.5)
    ax.set_title("Daily Revenue: Actual vs Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_DIR / "forecast_actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    print("\nDemand forecasting completed!")
    print(f"Training days: {len(train)}")
    print(f"Test days: {len(test)}")
    print(f"Test period: {test.index.min().date()} to {test.index.max().date()}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MAPE on nonzero-revenue days: {mape:.2f}%")
    print(f"7-day naive baseline MAE: {baseline_mae:,.2f}")
    print(f"Metrics saved: {REPORT_DIR / 'forecast_metrics.csv'}")
    print(f"Results saved: {REPORT_DIR / 'forecast_test_results.csv'}")


if __name__ == "__main__":
    run_forecasting()
