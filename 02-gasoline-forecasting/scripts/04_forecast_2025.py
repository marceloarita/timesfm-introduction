"""
04_forecast_2025.py
────────────────────
Out-of-sample validation: TimesFM vs XGBoost A on 2025 data.

Key question: does the winner from 2024 (XGBoost A) generalize to 2025?

XGBoost is retrained on 2013-2024 so that both models see the same data horizon
(TimesFM uses the last 156 weeks of context, which naturally includes 2024).

Regime analysis: compares demand patterns in 2024 vs 2025 to explain why
one model generalizes better than the other.

Outputs:
  charts/forecast_2025_vs_actual.png
  charts/forecast_2025_regime_analysis.png
  charts/forecast_2025_error_distribution.png
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import xgboost as xgb
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils import eval_table, mae

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#F1993A"
GREEN  = "#27AE60"
GRAY   = "#95A5A6"

BASE_DIR    = Path(__file__).parents[1]
DATA_PATH   = BASE_DIR / "data/raw/gasoline_weekly.csv"
PRICE_PATH  = BASE_DIR / "data/raw/gasoline_price_weekly.csv"
CHARTS_PATH = BASE_DIR / "charts"

CONTEXT_LEN   = 156
HORIZON       = 1
FORECAST_YEAR = 2025
TRAIN_END     = FORECAST_YEAR - 1   # 2024

# ── 1. Load data ───────────────────────────────────────────────────────────────
load_dotenv()
demand = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"])

print(f"Demand data: {demand['date'].min().date()} → {demand['date'].max().date()}")
n_2025 = (demand["date"].dt.year == FORECAST_YEAR).sum()
if n_2025 == 0:
    raise ValueError(f"No {FORECAST_YEAR} data in {DATA_PATH}. Re-run 00_download.py.")
print(f"  {FORECAST_YEAR} weeks available: {n_2025}")

values = demand["kbpd"].to_numpy(dtype=float)
dates  = demand["date"]

# ── 2. Merge demand + price ────────────────────────────────────────────────────
df = pd.merge_asof(
    demand.sort_values("date"),
    price_df.sort_values("date").rename(columns={"date": "price_date"}),
    left_on="date", right_on="price_date",
    tolerance=pd.Timedelta("7 days"),
    direction="nearest",
).drop(columns="price_date").sort_values("date").reset_index(drop=True)

# ── 3. Feature engineering ─────────────────────────────────────────────────────
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"]        = df["date"].dt.month
df["quarter"]      = df["date"].dt.quarter
df["year"]         = df["date"].dt.year
df["is_summer"]    = df["week_of_year"].between(20, 35).astype(int)
df["covid"]        = (df["year"] == 2020).astype(int)

for lag in [1, 4, 52, 104]:
    df[f"lag_{lag}"] = df["kbpd"].shift(lag)

df["rolling_4w_mean"]  = df["kbpd"].shift(1).rolling(4).mean()
df["rolling_13w_mean"] = df["kbpd"].shift(1).rolling(13).mean()
df["rolling_4w_std"]   = df["kbpd"].shift(1).rolling(4).std()

df["price_lag_1"]  = df["price_usd"].shift(1)
df["price_lag_4"]  = df["price_usd"].shift(4)
df["price_chg_4w"] = df["price_usd"] - df["price_usd"].shift(4)
df["price_vs_52w"] = df["price_usd"] - df["price_usd"].shift(52)

df = df.dropna().reset_index(drop=True)

FEATURES = [
    "week_of_year", "month", "quarter", "year", "is_summer", "covid",
    "lag_1", "lag_4", "lag_52", "lag_104",
    "rolling_4w_mean", "rolling_13w_mean", "rolling_4w_std",
    "price_lag_1", "price_lag_4", "price_chg_4w", "price_vs_52w",
]

# ── 4. XGBoost A — retrain on 2013-2024, predict 2025 ─────────────────────────
print(f"\nRetraining XGBoost on 2013–{TRAIN_END}...")
train = df[df["year"].between(2013, TRAIN_END)].copy()
test  = df[df["year"] == FORECAST_YEAR].copy()
if test.empty:
    raise ValueError(f"No rows for {FORECAST_YEAR} after feature engineering.")

weights = np.where(train["covid"] == 1, 0.3, 1.0)
model_xgb = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
model_xgb.fit(train[FEATURES], train["kbpd"], sample_weight=weights)
pred_xgb    = model_xgb.predict(test[FEATURES])
actual_2025 = test["kbpd"].values
naive_2025  = test["lag_52"].values
print(f"  Train: {len(train)} weeks  |  Test: {len(test)} weeks")

# ── 5. TimesFM — walk-forward on 2025 ─────────────────────────────────────────
print("Running TimesFM walk-forward on 2025...")
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

tfm = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", torch_compile=False
)
tfm.compile(ForecastConfig(max_context=CONTEXT_LEN, max_horizon=128, per_core_batch_size=1))

positions_2025 = demand[demand["date"].dt.year == FORECAST_YEAR].index.tolist()
tfm_records = []
for i, pos in enumerate(positions_2025):
    if pos < CONTEXT_LEN + 52:
        continue
    ctx   = values[pos - CONTEXT_LEN : pos]
    pf, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
    tfm_records.append({"date": dates[pos], "timesfm": float(pf[0][0])})
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(positions_2025)} done")

tfm_df = pd.DataFrame(tfm_records)
print(f"TimesFM: {len(tfm_df)} weeks predicted.")

# ── 6. Align predictions on common dates ──────────────────────────────────────
results = (
    test[["date", "kbpd", "lag_52"]]
    .rename(columns={"kbpd": "actual", "lag_52": "naive"})
    .assign(xgb_a=pred_xgb)
    .merge(tfm_df, on="date", how="inner")
    .reset_index(drop=True)
)

actual  = results["actual"].values
naive   = results["naive"].values
xgb_a   = results["xgb_a"].values
timesfm = results["timesfm"].values

# ── 7. Evaluation table ────────────────────────────────────────────────────────
eval_table(
    [
        ("Naive lag-52",                         naive,   actual, naive),
        ("TimesFM exp_156w",                     timesfm, actual, naive),
        (f"XGBoost A (train 2013–{TRAIN_END})",  xgb_a,   actual, naive),
    ],
    title=f"Out-of-Sample Validation ({FORECAST_YEAR})",
)

# ── 8. Chart — Forecast vs Actual ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(results["date"], actual,  color="black", linewidth=2.0, label="Actual 2025",       zorder=5)
ax.plot(results["date"], naive,   color=GRAY,    linewidth=1.2, linestyle="--", label="Naive lag-52")
ax.plot(results["date"], timesfm, color=BLUE,    linewidth=1.6, label="TimesFM exp_156w")
ax.plot(results["date"], xgb_a,   color=GREEN,   linewidth=1.6, label=f"XGBoost A (2013–{TRAIN_END})")

xgb_scaled = mae(xgb_a, actual) / mae(naive, actual)
tfm_scaled = mae(timesfm, actual) / mae(naive, actual)
ax.text(0.01, 0.04,
        f"Scaled MAE — TimesFM: {tfm_scaled:.3f}  |  XGBoost: {xgb_scaled:.3f}",
        transform=ax.transAxes, fontsize=10, color=GRAY, verticalalignment="bottom")

ax.set_title(f"U.S. Gasoline Demand — {FORECAST_YEAR} Out-of-Sample Forecast", fontsize=16)
ax.set_ylabel("Thousand Barrels/Day", fontsize=13)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=11, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_2025_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 9. Regime analysis — why does XGBoost struggle in 2025? ───────────────────
# If 2025 diverges meaningfully from 2024, those features carry a misleading signal.

df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
pivot = (
    df[df["year"].isin([2023, 2024, 2025])]
    .pivot_table(index="week_of_year", columns="year", values="kbpd")
)

# Deviation of 2025 from 2024 (what lag_52 anchors on)
deviation_2025_vs_2024 = (pivot[2025] - pivot[2024]).dropna()
mae_naive_2025 = mae(naive, actual)

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# Top: YoY overlay 2023-2025
for yr, color, lw in [(2023, GRAY, 1.0), (2024, ORANGE, 1.6), (2025, BLUE, 2.0)]:
    if yr in pivot.columns:
        axes[0].plot(pivot.index, pivot[yr], color=color, linewidth=lw, label=str(yr))
axes[0].set_title("Year-over-Year Comparison (2023–2025)", fontsize=13)
axes[0].set_ylabel("Thousand Barrels/Day", fontsize=11)
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
axes[0].legend(fontsize=11)
sns.despine(ax=axes[0])

# Bottom: 2025 deviation from 2024 (= error in lag_52 naive)
axes[1].bar(deviation_2025_vs_2024.index, deviation_2025_vs_2024.values,
            color=[ORANGE if v < 0 else BLUE for v in deviation_2025_vs_2024.values],
            width=0.8)
axes[1].axhline(0, color="black", linewidth=0.9)
axes[1].set_title("2025 Demand vs 2024 (same week) — Regime Shift Signal", fontsize=13)
axes[1].set_xlabel("ISO Week", fontsize=11)
axes[1].set_ylabel("Δ kbpd  (2025 − 2024)", fontsize=11)
axes[1].text(0.01, 0.05,
             "Orange = 2025 below 2024  |  Blue = 2025 above 2024\n"
             "XGBoost learned momentum + seasonality patterns from 2013–2024.\n"
             "When those relationships shift in 2025, TimesFM (zero-shot) adapts better.",
             transform=axes[1].transAxes, fontsize=9, color=GRAY, verticalalignment="bottom")
sns.despine(ax=axes[1])

plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_2025_regime_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 10. Chart — Error distribution comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (name, pred, color) in zip(axes, [
    ("TimesFM exp_156w",                  timesfm, BLUE),
    (f"XGBoost A (2013–{TRAIN_END})",     xgb_a,   GREEN),
]):
    errors = pred - actual
    ax.hist(errors, bins=18, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(errors.mean(), color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Mean = {errors.mean():+.0f}")
    ax.set_title(f"{name}\nMAE = {mae(pred, actual):.0f} kbpd", fontsize=12)
    ax.set_xlabel("Error (Predicted − Actual, kbpd)", fontsize=11)
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    sns.despine(ax=ax)
axes[0].set_ylabel("Count", fontsize=11)
fig.suptitle(f"Forecast Error Distribution — {FORECAST_YEAR}", fontsize=14)
plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_2025_error_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nCharts saved to {CHARTS_PATH}")
