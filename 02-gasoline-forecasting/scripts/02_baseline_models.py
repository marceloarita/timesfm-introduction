"""
02_baseline_models.py
─────────────────────
Baseline model comparison for U.S. gasoline demand (2024 test set).

Models:
  1. Naive lag-52         — same week last year
  2. TimesFM exp_156w     — zero-shot, context sweep 52–416 weeks (elbow)
  3. XGBoost A (direct)   — supervised, train 2013-2023, feature importance

Outputs:
  charts/elbow_context_length.png
  charts/baseline_feature_importance.png
  data/processed/walk_forward_results.csv   (TimesFM sweep — cached)
  data/processed/baseline_results_2024.csv  (aligned predictions for script 03)
  data/models/xgb_baseline.joblib
"""

import sys
from pathlib import Path
import os
import requests
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
GRAY   = "#95A5A6"

BASE_DIR     = Path(__file__).parents[1]
DATA_PATH    = BASE_DIR / "data/raw/gasoline_weekly.csv"
PRICE_PATH   = BASE_DIR / "data/raw/gasoline_price_weekly.csv"
WF_PATH      = BASE_DIR / "data/processed/walk_forward_results.csv"
RESULTS_PATH = BASE_DIR / "data/processed/baseline_results_2024.csv"
CHARTS_PATH  = BASE_DIR / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

CONTEXT_LEN = 156
HORIZON     = 1
TEST_YEAR   = 2024

# ── 1. Load demand ─────────────────────────────────────────────────────────────
demand = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
values = demand["kbpd"].to_numpy(dtype=float)
dates  = demand["date"]
print(f"Demand: {len(demand)} weeks  ({demand['date'].min().date()} → {demand['date'].max().date()})")

# ── 2. Load / download price data ──────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("EIA_API_KEY", "")

if not PRICE_PATH.exists():
    print("Downloading price data from EIA...")
    url = (
        "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
        f"?api_key={API_KEY}"
        "&frequency=weekly"
        "&data[0]=value"
        "&facets[product][]=EPMR"
        "&facets[duoarea][]=NUS"
        "&start=2010-01-01"
        "&end=2025-12-31"
        "&sort[0][column]=period"
        "&sort[0][direction]=asc"
        "&length=5000"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    records = resp.json()["response"]["data"]
    if not records:
        raise ValueError("EIA price API returned 0 records.")
    price_df = pd.DataFrame(records)[["period", "value"]]
    price_df.columns = ["date", "price_usd"]
    price_df["date"]      = pd.to_datetime(price_df["date"])
    price_df["price_usd"] = pd.to_numeric(price_df["price_usd"], errors="coerce")
    price_df = price_df.sort_values("date").reset_index(drop=True)
    price_df.to_csv(PRICE_PATH, index=False)
    print(f"  Saved → {PRICE_PATH} ({len(price_df)} weeks)")
else:
    price_df = pd.read_csv(PRICE_PATH, parse_dates=["date"])
    print(f"Price data loaded from cache ({len(price_df)} weeks)")

# ── 3. Merge demand + price (nearest date, ±7 days) ────────────────────────────
df = pd.merge_asof(
    demand.sort_values("date"),
    price_df.sort_values("date").rename(columns={"date": "price_date"}),
    left_on="date", right_on="price_date",
    tolerance=pd.Timedelta("7 days"),
    direction="nearest",
).drop(columns="price_date").sort_values("date").reset_index(drop=True)

print(f"Merged rows: {len(df)}  |  Price NAs: {df['price_usd'].isna().sum()}")

# ── 4. Feature engineering ─────────────────────────────────────────────────────
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

# ── 5. TimesFM context sweep (load cache or run) ───────────────────────────────
if WF_PATH.exists():
    wf_results = pd.read_csv(WF_PATH, parse_dates=["date"])
    print(f"\nTimesFM sweep loaded from cache ({len(wf_results)} records, "
          f"{wf_results['experiment'].nunique()} experiments)")
else:
    print("\nRunning TimesFM context sweep (no cache found)...")
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    EXPERIMENTS    = [(f"exp_{w}w", w) for w in range(52, 417, 52)]
    test_positions = demand[demand["date"].dt.year == TEST_YEAR].index.tolist()

    records = []
    for exp_name, ctx_len in EXPERIMENTS:
        tfm.compile(ForecastConfig(max_context=ctx_len, max_horizon=128, per_core_batch_size=1))
        print(f"  [{exp_name}] {len(test_positions)} steps")
        for pos in test_positions:
            if pos < ctx_len + 52:
                continue
            ctx   = values[pos - ctx_len : pos]
            pf, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
            records.append({
                "experiment": exp_name, "context_len": ctx_len,
                "date":       dates[pos],
                "actual":     values[pos],
                "predicted":  float(pf[0][0]),
                "naive":      values[pos - 52],
            })

    wf_results = pd.DataFrame(records)
    WF_PATH.parent.mkdir(exist_ok=True)
    wf_results.to_csv(WF_PATH, index=False)
    print(f"  Saved → {WF_PATH}")

# ── 6. Elbow plot — Scaled MAE vs context length ───────────────────────────────
summary_rows = []
for ctx_len, grp in wf_results.groupby("context_len"):
    m         = (grp["predicted"] - grp["actual"]).abs().mean()
    naive_mae = (grp["naive"]     - grp["actual"]).abs().mean()
    summary_rows.append({
        "ctx_len":    ctx_len,
        "exp_name":   grp["experiment"].iloc[0],
        "scaled_mae": m / naive_mae,
    })
summary_df = pd.DataFrame(summary_rows)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(summary_df["ctx_len"], summary_df["scaled_mae"],
        color=BLUE, linewidth=2, marker="o", markersize=7)
for _, row in summary_df.iterrows():
    ax.annotate(f"{row['scaled_mae']:.3f}",
                xy=(row["ctx_len"], row["scaled_mae"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=10, color=BLUE)
ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1.2, label="Naive lag-52 (= 1.0)")
ax.set_title("TimesFM — Scaled MAE vs Context Length (2024 test)", fontsize=15)
ax.set_xlabel("Context Length (weeks)", fontsize=13)
ax.set_ylabel("Scaled MAE", fontsize=13)
ax.set_xticks(summary_df["ctx_len"])
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "elbow_context_length.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7. XGBoost A — direct demand prediction ────────────────────────────────────
print("\nTraining XGBoost A (direct demand prediction)...")
train_a   = df[df["year"].between(2013, 2023)].copy()
test_a    = df[df["year"] == TEST_YEAR].copy()
weights_a = np.where(train_a["covid"] == 1, 0.3, 1.0)

model_a = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    importance_type="gain",
    random_state=42, verbosity=0,
)
model_a.fit(train_a[FEATURES], train_a["kbpd"], sample_weight=weights_a)
pred_xgb_2024 = model_a.predict(test_a[FEATURES])
print(f"  Train: {len(train_a)} weeks (2013–2023)  |  Test: {len(test_a)} weeks (2024)")

save_model(model_a, "xgb_baseline")

# ── 8. Feature importance chart ────────────────────────────────────────────────
importance = (
    pd.Series(model_a.feature_importances_, index=FEATURES)
    .sort_values(ascending=True)
)

fig, ax = plt.subplots(figsize=(9, 6))
colors = [ORANGE if importance[f] == importance.max() else BLUE for f in importance.index]
ax.barh(importance.index, importance.values, color=colors, height=0.65)
ax.set_title("XGBoost A — Feature Importance (Gain)", fontsize=14)
ax.set_xlabel("Normalized Gain", fontsize=12)
ax.tick_params(labelsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "baseline_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nTop 5 features by gain:")
for feat, val in importance.sort_values(ascending=False).head(5).items():
    print(f"  {feat:<25} {val:.4f}")

# ── 9. Align predictions for evaluation ───────────────────────────────────────
# Inner join TimesFM exp_156w + XGBoost on date
tfm_156 = (
    wf_results[wf_results["experiment"] == f"exp_{CONTEXT_LEN}w"]
    [["date", "predicted", "naive"]]
    .rename(columns={"predicted": "timesfm"})
)

eval_df = (
    test_a[["date", "kbpd", "lag_52"]]
    .rename(columns={"kbpd": "actual", "lag_52": "naive"})
    .assign(xgb_a=pred_xgb_2024)
    .merge(tfm_156[["date", "timesfm"]], on="date", how="inner")
)

actual  = eval_df["actual"].values
naive   = eval_df["naive"].values
timesfm = eval_df["timesfm"].values
xgb_a   = eval_df["xgb_a"].values

eval_table(
    [
        ("Naive lag-52",       naive,   actual, naive),
        ("TimesFM exp_156w",   timesfm, actual, naive),
        ("XGBoost A (direct)", xgb_a,   actual, naive),
    ],
    title="Baseline Model Evaluation (2024)",
)

# ── 10. Save for script 03 ─────────────────────────────────────────────────────
eval_df.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved → {RESULTS_PATH}  ({len(eval_df)} weeks)")
