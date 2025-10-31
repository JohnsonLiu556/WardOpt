"""
Reads 'hospital_capacity_tidy.csv' and generates 'predictions_monthly.csv'
for the latest month. Includes naive, moving-average, and random forest models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(".")  
INPUT_FILE = DATA_DIR / "Hospital_Capacity_Data.csv"
OUTPUT_FILE = DATA_DIR / "Monthly_Predictions.csv"

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
df = df.sort_values(["trust", "dept", "date"])

df["patients_lag1"] = df.groupby(["trust", "dept"])["patients"].shift(1)
df["beds_lag1"] = df.groupby(["trust", "dept"])["beds"].shift(1)
df["occ_lag1"] = df["patients_lag1"] / df["beds_lag1"].replace(0, np.nan)
df["patients_ma3"] = (
    df.groupby(["trust", "dept"])["patients"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .shift(1)
)
df["month"] = df["date"].dt.month
df["m_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["m_cos"] = np.cos(2 * np.pi * df["month"] / 12)
data = df.dropna(subset=["patients_lag1", "beds_lag1", "occ_lag1", "patients_ma3"])

last_date = data["date"].max()
train = data[data["date"] < last_date]
test = data[data["date"] == last_date]

test["pred_naive"] = test["patients_lag1"]
test["pred_ma3"] = test["patients_ma3"]

rf = RandomForestRegressor(n_estimators=300, random_state=42)
features = ["patients_lag1", "beds_lag1", "occ_lag1", "m_sin", "m_cos"]
rf.fit(train[features], train["patients"])
test["pred_rf"] = rf.predict(test[features])

def evaluate(true, pred):
    return {
        "MAE": round(mean_absolute_error(true, pred), 2),
        "R2": round(r2_score(true, pred), 3)
    }

results = {
    "Naive": evaluate(test["patients"], test["pred_naive"]),
    "MA(3)": evaluate(test["patients"], test["pred_ma3"]),
    "RandomForest": evaluate(test["patients"], test["pred_rf"])
}

print("\nModel Results (lower MAE is better):")
for name, score in results.items():
    print(f"{name:15s} MAE={score['MAE']:6.2f} | R2={score['R2']:5.3f}")

best_model = min(results, key=lambda k: results[k]["MAE"])
print(f"\n Best model: {best_model}\n")

pred = test[["date", "trust", "dept", "beds", "patients"]].copy()
name_map = {"Naive": "pred_naive", "MA(3)": "pred_ma3", "RandomForest": "pred_rf"}
pred["pred_patients"] = test[name_map[best_model]]

pred.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to: {OUTPUT_FILE.resolve()}")

# 7. Plot by department
agg = pred.groupby("dept", as_index=False)[["patients", "pred_patients"]].sum()
for _, row in agg.iterrows():
    dept = row["dept"]
    hist = df[df["dept"] == dept].groupby("date", as_index=False)[["patients"]].sum()
    plt.figure()
    plt.plot(hist["date"], hist["patients"], marker="o", label="actual")
    plt.scatter([last_date], [row["pred_patients"]], color="red", label="forecast")
    plt.title(f"{dept} — Forecast for {last_date.date()}")
    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.show()
