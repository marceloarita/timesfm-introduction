"""
03_ensemble.py
──────────────
Ensemble: TimesFM + best residual correction model (2024 test set).

Strategy:
  TimesFM makes a base forecast. A residual correction model learns to predict
  the systematic errors that TimesFM makes. Final prediction = TimesFM + correction.

  Four candidate models are evaluated on their ability to predict TimesFM residuals
  (R², Pearson r) on the 2024 test set. The best model is selected to build the
  final ensemble.

  Candidates: XGBoost, LightGBM, Lasso (CV), ElasticNet (CV)
  Training set: 2014-2023 TimesFM residuals
  Test set: 2024

Outputs:
  charts/ensemble_residual_r2_comparison.png
  charts/ensemble_residual_scatter.png
  charts/ensemble_forecast_2024.png
  charts/ensemble_scaled_mae_bar.png
  data/processed/ensemble_results_2024.csv
  data/models/best_residual_model.joblib
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from utils import save_model, eval_table, mae

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
ORANGE = "#F1993A"
GREEN  = "#27AE60"
PURPLE = "#8E44AD"
GRAY   = "#95A5A6"

BASE_DIR        = Path(__file__).parents[1]
DATA_PATH       = BASE_DIR / "data/raw/gasoline_weekly.csv"
PRICE_PATH      = BASE_DIR / "data/raw/gasoline_price_weekly.csv"
BASELINE_PATH   = BASE_DIR / "data/processed/baseline_results_2024.csv"
RESIDUALS_CACHE = BASE_DIR / "data/processed/residual_walk_forward.csv"
RESULTS_PATH    = BASE_DIR / "data/processed/ensemble_results_2024.csv"
CHARTS_PATH     = BASE_DIR / "charts"

CONTEXT_LEN    = 156
HORIZON        = 1
RESIDUAL_YEARS = list(range(2014, 2024))
TEST_YEAR      = 2024

# ── 1. Load baseline TimesFM predictions (script 02 output) ───────────────────
if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"{BASELINE_PATH} not found.\nRun 02_baseline_models.py first."
    )

base = pd.read_csv(BASELINE_PATH, parse_dates=["date"])
actual_2024  = base["actual"].values
naive_2024   = base["naive"].values
timesfm_2024 = base["timesfm"].values
dates_2024   = base["date"].values

print(f"Baseline: {len(base)} weeks  ({base['date'].min().date()} -> {base['date'].max().date()})")

# ── 2. Load demand + price, build features ────────────────────────────────────
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

for lag in [1, 4, 52]:
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
    "week_of_year", "month", "quarter", "is_summer",
    "lag_1", "lag_4", "lag_52",
    "rolling_4w_mean", "rolling_13w_mean", "rolling_4w_std",
    "price_lag_1", "price_lag_4", "price_chg_4w", "price_vs_52w",
]

# ── 3. TimesFM walk-forward on residual years (2014-2023) — cached ────────────
if RESIDUALS_CACHE.exists():
    residuals_df = pd.read_csv(RESIDUALS_CACHE, parse_dates=["date"])
    print(f"\nResidual walk-forward loaded from cache ({len(residuals_df)} records)")
else:
    print(f"\nRunning TimesFM walk-forward on {RESIDUAL_YEARS[0]}-{RESIDUAL_YEARS[-1]}...")
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
    residuals_df.to_csv(RESIDUALS_CACHE, index=False)
    print(f"  Saved -> {RESIDUALS_CACHE}")

test_residuals = pd.DataFrame({
    "date":         dates_2024,
    "actual":       actual_2024,
    "timesfm_pred": timesfm_2024,
    "residual":     actual_2024 - timesfm_2024,
})
residuals_df = pd.concat([residuals_df, test_residuals], ignore_index=True)

df_feat      = df[["date"] + FEATURES].copy()
residuals_df = residuals_df.merge(df_feat, on="date", how="left")
print(f"  Total records: {len(residuals_df)}  (train + test)")

train_r = residuals_df[residuals_df["date"].dt.year.isin(RESIDUAL_YEARS)].dropna()
test_r  = residuals_df[residuals_df["date"].dt.year == TEST_YEAR].dropna()
print(f"  Train: {len(train_r)} weeks  |  Test: {len(test_r)} weeks")

# ── 4. Residual model candidates ──────────────────────────────────────────────
X_train = train_r[FEATURES].values
y_train = train_r["residual"].values
X_test  = test_r[FEATURES].values
y_test  = test_r["residual"].values

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

def residual_r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

def pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])

candidates = {}
print("\nFitting residual model candidates...")

m_xgb = xgb.XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
m_xgb.fit(X_train, y_train)
p_xgb = m_xgb.predict(X_test)
candidates["XGBoost"] = {"model": m_xgb, "pred": p_xgb,
                          "r2": residual_r2(y_test, p_xgb),
                          "pearson_r": pearson_r(y_test, p_xgb)}

m_lgb = lgb.LGBMRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
)
m_lgb.fit(X_train, y_train)
p_lgb = m_lgb.predict(X_test)
candidates["LightGBM"] = {"model": m_lgb, "pred": p_lgb,
                           "r2": residual_r2(y_test, p_lgb),
                           "pearson_r": pearson_r(y_test, p_lgb)}

m_lasso = LassoCV(cv=5, max_iter=5000, random_state=42)
m_lasso.fit(X_train_sc, y_train)
p_lasso = m_lasso.predict(X_test_sc)
candidates["Lasso (CV)"] = {"model": m_lasso, "pred": p_lasso,
                             "r2": residual_r2(y_test, p_lasso),
                             "pearson_r": pearson_r(y_test, p_lasso)}

m_enet = ElasticNetCV(cv=5, max_iter=5000, random_state=42)
m_enet.fit(X_train_sc, y_train)
p_enet = m_enet.predict(X_test_sc)
candidates["ElasticNet (CV)"] = {"model": m_enet, "pred": p_enet,
                                  "r2": residual_r2(y_test, p_enet),
                                  "pearson_r": pearson_r(y_test, p_enet)}

# ── 5. Comparison table ────────────────────────────────────────────────────────
print(f"\n-- Residual Model Comparison (2024 test set) --")
print(f"{'Model':<20} {'R2':>8} {'Pearson r':>12}")
print("-" * 44)
for name, info in candidates.items():
    print(f"{name:<20} {info['r2']:>8.3f} {info['pearson_r']:>12.3f}")
print("-" * 44)

best_name = max(candidates, key=lambda k: candidates[k]["r2"])
best      = candidates[best_name]
print(f"\nBest residual model: {best_name}  (R2 = {best['r2']:.3f})")

# ── 6. Chart — R2 comparison bar ──────────────────────────────────────────────
model_names = list(candidates.keys())
r2_values   = [candidates[n]["r2"] for n in model_names]
bar_colors  = [ORANGE if n == best_name else BLUE for n in model_names]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(model_names, r2_values, color=bar_colors, width=0.45, edgecolor="white")
ax.axhline(0, color="black", linewidth=0.9)

for bar, val in zip(bars, r2_values):
    x = bar.get_x() + bar.get_width() / 2
    if val >= 0:
        ax.text(x, val + 0.004, f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    else:
        ax.text(x, val - 0.004, f"{val:.3f}", ha="center", va="top",
                fontsize=12, fontweight="bold")

ymin = min(r2_values) - 0.05
ymax = max(r2_values) + 0.03
ax.set_ylim(ymin, ymax)
ax.set_title("Residual Model Candidates -- R2 on 2024 Test Set\n(higher = better residual prediction)",
             fontsize=13)
ax.set_ylabel("R2  (variance of TimesFM errors explained)", fontsize=12)
ax.tick_params(axis="x", labelsize=11, pad=8)
ax.tick_params(axis="y", labelsize=10)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_residual_r2_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7. Residual scatter — best model ──────────────────────────────────────────
pred_resid   = best["pred"]
actual_resid = y_test
n_helped = int((
    np.abs(test_r["timesfm_pred"].values + pred_resid - test_r["actual"].values)
    < np.abs(actual_resid)
).sum())

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
    f"{best_name} -- Best Residual Model (2024 Test)\n"
    f"Pearson r = {best['pearson_r']:+.3f}  |  R2 = {best['r2']:.3f}"
    f"  |  Helped {n_helped}/{len(actual_resid)} weeks",
    fontsize=12,
)
ax.legend(fontsize=10)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_residual_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8. Build final ensemble ────────────────────────────────────────────────────
save_model(best["model"], "best_residual_model")
ensemble_2024 = test_r["timesfm_pred"].values + pred_resid

# ── 9. Evaluation table ────────────────────────────────────────────────────────
eval_table(
    [
        ("Naive lag-52",                           naive_2024,    actual_2024, naive_2024),
        ("TimesFM exp_156w",                       timesfm_2024,  actual_2024, naive_2024),
        (f"TFM + {best_name} Residual (ensemble)", ensemble_2024, actual_2024, naive_2024),
    ],
    title="Model Evaluation (2024)",
)

# ── 10. Chart — Forecast vs Actual ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(base["date"], actual_2024,   color="black",  linewidth=2.0, label="Actual",         zorder=5)
ax.plot(base["date"], naive_2024,    color=GRAY,     linewidth=1.1, linestyle="--", label="Naive lag-52")
ax.plot(base["date"], timesfm_2024,  color=BLUE,     linewidth=1.4, label="TimesFM exp_156w")
ax.plot(base["date"], ensemble_2024, color=PURPLE,   linewidth=1.6, linestyle="-.",
        label=f"TFM + {best_name} Residual")

ax.set_title("U.S. Gasoline Demand -- Ensemble vs TimesFM (2024)", fontsize=16)
ax.set_ylabel("Thousand Barrels/Day", fontsize=13)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=10, ncol=2, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_forecast_2024.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 11. Chart — Scaled MAE bar ────────────────────────────────────────────────
naive_mae_val = mae(naive_2024, actual_2024)
bar_data = [
    ("Naive",               naive_2024,    GRAY),
    ("TimesFM\nexp_156w",   timesfm_2024,  BLUE),
    (f"TFM +\n{best_name}", ensemble_2024, PURPLE),
]
bar_names  = [r[0] for r in bar_data]
bar_colors = [r[2] for r in bar_data]
bar_scaled = [mae(r[1], actual_2024) / naive_mae_val for r in bar_data]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(bar_names, bar_scaled, color=bar_colors, width=0.5, edgecolor="white")
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.2, label="Naive baseline (= 1.0)")
for bar, val in zip(bars, bar_scaled):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_title("Scaled MAE -- Model Comparison (2024 test set)", fontsize=15)
ax.set_ylabel("Scaled MAE  (lower = better)", fontsize=13)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_scaled_mae_bar.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 12. Save for script 04 ────────────────────────────────────────────────────
out = pd.DataFrame({
    "date":       dates_2024,
    "actual":     actual_2024,
    "naive":      naive_2024,
    "timesfm":    timesfm_2024,
    "ensemble":   ensemble_2024,
    "best_model": best_name,
})
out.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved -> {RESULTS_PATH}")
print(f"Best residual model: {best_name}")
