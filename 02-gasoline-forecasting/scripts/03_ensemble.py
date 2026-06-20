"""
03_ensemble.py
──────────────
Ensemble: TimesFM + XGBoost Residual Correction (2024 test set).

Strategy:
  TimesFM makes a base forecast. A second XGBoost model learns to predict
  the systematic errors (residuals) that TimesFM makes, using calendar and
  price signals as features. The final prediction = TimesFM + predicted residual.

  Training set for XGBoost residual: 2014-2023 (TimesFM errors on those years).
  Test set: 2024.

Residual diagnostic: evaluates whether the correction model adds real signal
(Pearson r, R², weeks helped vs hurt) and visualises actual vs predicted residuals.

Conclusion: XGBoost A (direct) wins on 2024 — the ensemble does not improve
over a direct supervised model on an in-sample regime.

Outputs:
  charts/ensemble_forecast_2024.png
  charts/ensemble_scaled_mae_bar.png
  charts/ensemble_residual_scatter.png
  charts/ensemble_residual_timeseries.png
  data/processed/ensemble_results_2024.csv
  data/models/xgb_residual.joblib
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
from utils import save_model, eval_table, mae

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#F1993A"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
GRAY   = "#95A5A6"

BASE_DIR      = Path(__file__).parents[1]
DATA_PATH     = BASE_DIR / "data/raw/gasoline_weekly.csv"
PRICE_PATH    = BASE_DIR / "data/raw/gasoline_price_weekly.csv"
BASELINE_PATH = BASE_DIR / "data/processed/baseline_results_2024.csv"
RESULTS_PATH  = BASE_DIR / "data/processed/ensemble_results_2024.csv"
CHARTS_PATH   = BASE_DIR / "charts"

CONTEXT_LEN    = 156
HORIZON        = 1
RESIDUAL_YEARS = list(range(2014, 2024))   # 2014-2023
TEST_YEAR      = 2024

# ── 1. Load baseline predictions (script 02 output) ───────────────────────────
if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"{BASELINE_PATH} not found.\nRun 02_baseline_models.py first."
    )

base = pd.read_csv(BASELINE_PATH, parse_dates=["date"])
actual_2024  = base["actual"].values
naive_2024   = base["naive"].values
timesfm_2024 = base["timesfm"].values
xgb_a_2024   = base["xgb_a"].values
dates_2024   = base["date"].values

print(f"Baseline results: {len(base)} weeks  ({base['date'].min().date()} → {base['date'].max().date()})")

# ── 2. Load full demand + price for feature engineering ────────────────────────
demand = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
values = demand["kbpd"].to_numpy(dtype=float)
dates  = demand["date"]

load_dotenv()
price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"])

df = pd.merge_asof(
    demand.sort_values("date"),
    price_df.sort_values("date").rename(columns={"date": "price_date"}),
    left_on="date", right_on="price_date",
    tolerance=pd.Timedelta("7 days"),
    direction="nearest",
).drop(columns="price_date").sort_values("date").reset_index(drop=True)

df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["month"]        = df["date"].dt.month
df["quarter"]      = df["date"].dt.quarter
df["year"]         = df["date"].dt.year
df["is_summer"]    = df["week_of_year"].between(20, 35).astype(int)

df["price_lag_1"]  = df["price_usd"].shift(1)
df["price_lag_4"]  = df["price_usd"].shift(4)
df["price_chg_4w"] = df["price_usd"] - df["price_usd"].shift(4)
df["price_vs_52w"] = df["price_usd"] - df["price_usd"].shift(52)

df = df.dropna().reset_index(drop=True)

FEATURES_RESIDUAL = [
    "week_of_year", "month", "quarter", "is_summer",
    "price_lag_1", "price_lag_4", "price_chg_4w", "price_vs_52w",
]

# ── 3. TimesFM walk-forward on residual years (training data for residual XGB) ─
print(f"\nRunning TimesFM walk-forward on {RESIDUAL_YEARS[0]}–{RESIDUAL_YEARS[-1]}...")
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

tfm = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch", torch_compile=False
)
tfm.compile(ForecastConfig(max_context=CONTEXT_LEN, max_horizon=128, per_core_batch_size=1))

residual_records = []
for year in RESIDUAL_YEARS:
    positions = demand[demand["date"].dt.year == year].index.tolist()
    print(f"  {year}: {len(positions)} steps")
    for pos in positions:
        if pos < CONTEXT_LEN + 52:
            continue
        ctx    = values[pos - CONTEXT_LEN : pos]
        actual = values[pos]
        pf, _  = tfm.forecast(horizon=HORIZON, inputs=[ctx])
        pred   = float(pf[0][0])
        residual_records.append({
            "date":         dates[pos],
            "actual":       actual,
            "timesfm_pred": pred,
            "residual":     actual - pred,
        })

residuals_df = pd.DataFrame(residual_records).sort_values("date").reset_index(drop=True)

# Append 2024 rows (from baseline) to form the test set
test_residuals = pd.DataFrame({
    "date":         dates_2024,
    "actual":       actual_2024,
    "timesfm_pred": timesfm_2024,
    "residual":     actual_2024 - timesfm_2024,
})
residuals_df = pd.concat([residuals_df, test_residuals], ignore_index=True)

# Merge calendar + price features
df_feat = df[["date"] + FEATURES_RESIDUAL].copy()
residuals_df = residuals_df.merge(df_feat, on="date", how="left")
print(f"  Total records: {len(residuals_df)}  (train + test)")

# ── 4. XGBoost Residual — learns TimesFM's systematic errors ──────────────────
train_r = residuals_df[residuals_df["date"].dt.year.isin(RESIDUAL_YEARS)].dropna()
test_r  = residuals_df[residuals_df["date"].dt.year == TEST_YEAR].dropna()

model_r = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)
model_r.fit(train_r[FEATURES_RESIDUAL], train_r["residual"])
pred_residual    = model_r.predict(test_r[FEATURES_RESIDUAL])
ensemble_2024    = test_r["timesfm_pred"].values + pred_residual

save_model(model_r, "xgb_residual")
print(f"\nXGBoost Residual trained on {len(train_r)} weeks ({RESIDUAL_YEARS[0]}–{RESIDUAL_YEARS[-1]})")

# ── 5. Residual model diagnostic — signal or noise? ───────────────────────────
actual_resid = test_r["residual"].values      # TimesFM error on 2024
pred_resid   = pred_residual                   # XGBoost B's prediction of that error

corr = float(np.corrcoef(actual_resid, pred_resid)[0, 1])
ss_res = float(((actual_resid - pred_resid) ** 2).sum())
ss_tot = float(((actual_resid - actual_resid.mean()) ** 2).sum())
r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

mae_before = np.abs(actual_resid)
mae_after  = np.abs(test_r["timesfm_pred"].values + pred_resid - test_r["actual"].values)
n_helped   = int((mae_after < mae_before).sum())
n_hurt     = len(actual_resid) - n_helped

print(f"\nResidual model diagnostic (2024 test set):")
print(f"  Pearson r (actual vs predicted residual) : {corr:+.3f}")
print(f"  R²  of residual prediction               : {r2:.3f}")
print(f"  Correction helped : {n_helped}/{len(actual_resid)} weeks")
print(f"  Correction hurt   : {n_hurt}/{len(actual_resid)} weeks")

# Chart A — Scatter: actual residual vs predicted residual
lim = max(np.abs(actual_resid).max(), np.abs(pred_resid).max()) * 1.1
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(actual_resid, pred_resid, color=PURPLE, alpha=0.6, edgecolors="white", s=55, zorder=3)
ax.plot([-lim, lim], [-lim, lim], color=GRAY, linewidth=1.2, linestyle="--", label="Perfect correction")
ax.axhline(0, color="black", linewidth=0.7)
ax.axvline(0, color="black", linewidth=0.7)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Actual residual  (kbpd)", fontsize=12)
ax.set_ylabel("Predicted residual  (kbpd)", fontsize=12)
ax.set_title(
    f"XGBoost Residual Model — 2024 Test\n"
    f"Pearson r = {corr:+.3f}  |  R² = {r2:.3f}  |  Helped {n_helped}/{len(actual_resid)} weeks",
    fontsize=12,
)
ax.legend(fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_residual_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart B — Time series: actual vs predicted residual
dates_test = pd.to_datetime(test_r["date"].values)
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(dates_test, actual_resid, color=BLUE,   linewidth=1.6, label="Actual residual (TimesFM error)")
ax.plot(dates_test, pred_resid,   color=ORANGE, linewidth=1.6, linestyle="--",
        label="Predicted residual (XGBoost Residual)")
ax.fill_between(dates_test, actual_resid, pred_resid, alpha=0.12, color=PURPLE)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Residual: Actual vs Predicted — 2024 Test Set", fontsize=14)
ax.set_ylabel("Residual (kbpd)", fontsize=12)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_residual_timeseries.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 6. Evaluation table ────────────────────────────────────────────────────────
eval_table(
    [
        ("Naive lag-52",                     naive_2024,   actual_2024, naive_2024),
        ("TimesFM exp_156w",                 timesfm_2024, actual_2024, naive_2024),
        ("XGBoost A (direct)",               xgb_a_2024,   actual_2024, naive_2024),
        ("TFM + XGBoost Residual (ensemble)", ensemble_2024, actual_2024, naive_2024),
    ],
    title="Model Evaluation (2024)",
)

# ── 7. Chart — Forecast vs Actual ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(base["date"], actual_2024,    color="black",  linewidth=2.0, label="Actual",               zorder=5)
ax.plot(base["date"], naive_2024,     color=GRAY,     linewidth=1.1, linestyle="--", label="Naive lag-52")
ax.plot(base["date"], timesfm_2024,   color=BLUE,     linewidth=1.4, label="TimesFM exp_156w")
ax.plot(base["date"], xgb_a_2024,     color=GREEN,    linewidth=1.4, label="XGBoost A (direct)")
ax.plot(base["date"], ensemble_2024,  color=PURPLE,   linewidth=1.6, linestyle="-.",
        label="TFM + XGBoost Residual")

ax.set_title("U.S. Gasoline Demand — Model Comparison (2024)", fontsize=16)
ax.set_ylabel("Thousand Barrels/Day", fontsize=13)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=10, ncol=2, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_forecast_2024.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8. Chart — Scaled MAE bar chart ───────────────────────────────────────────
naive_mae_val = mae(naive_2024, actual_2024)
bar_data = [
    ("Naive",                     naive_2024,    GRAY),
    ("TimesFM\nexp_156w",         timesfm_2024,  BLUE),
    ("XGBoost A\n(direct)",       xgb_a_2024,    GREEN),
    ("TFM + XGBoost\nResidual",   ensemble_2024, PURPLE),
]
bar_names  = [r[0] for r in bar_data]
bar_colors = [r[2] for r in bar_data]
bar_scaled = [mae(r[1], actual_2024) / naive_mae_val for r in bar_data]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(bar_names, bar_scaled, color=bar_colors, width=0.5, edgecolor="white")
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.2, label="Naive baseline (= 1.0)")
for bar, val in zip(bars, bar_scaled):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_title("Scaled MAE — Model Comparison (2024 test set)", fontsize=15)
ax.set_ylabel("Scaled MAE  (lower = better)", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_scaled_mae_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 9. Save for script 04 ─────────────────────────────────────────────────────
out = pd.DataFrame({
    "date":     dates_2024,
    "actual":   actual_2024,
    "naive":    naive_2024,
    "timesfm":  timesfm_2024,
    "xgb_a":    xgb_a_2024,
    "ensemble": ensemble_2024,
})
out.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved → {RESULTS_PATH}")
