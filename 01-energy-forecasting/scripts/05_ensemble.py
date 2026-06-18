from pathlib import Path
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="muted")
BLUE   = "#147EC5"
GREEN  = "#AF4DED"
GRAY   = "#828A8AB3"
ORANGE = "#E87722"

WF_2017_PATH     = Path(__file__).parents[1] / "data/processed/walk_forward_2017.csv"
WF_2018_PATH     = Path(__file__).parents[1] / "data/processed/walk_forward_results.csv"
WEATHER_PATH     = Path(__file__).parents[1] / "data/processed/weather_philadelphia.csv"
CHARTS_PATH      = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

BASE_EXP = "exp3_8736h"

# Top features selected from SHAP analysis in 04_residual_model.py
WEATHER_COLS      = ["temperature", "apparent_temperature", "humidity"]
ENSEMBLE_FEATURES = ["temperature", "apparent_temperature", "humidity", "HDD", "CDD"]

EXPERIMENTS = [
    ("exp1_512h",  512),
    ("exp2_672h",  672),
    ("exp3_8736h", 8736),
]

###########################################
# Load data
###########################################
weather = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"]).set_index("datetime")

wf_2017 = pd.read_csv(WF_2017_PATH, parse_dates=["date"])
wf_2018 = pd.read_csv(WF_2018_PATH, parse_dates=["date"])

def attach_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df["residual"]  = df["actual"] - df["predicted"]
    df = df.join(weather[WEATHER_COLS], on="datetime")
    df["HDD"] = (18 - df["temperature"]).clip(lower=0)
    df["CDD"] = (df["temperature"] - 18).clip(lower=0)
    return df.dropna(subset=ENSEMBLE_FEATURES + ["residual"])

train_df = attach_weather(wf_2017)
test_base = wf_2018[wf_2018["experiment"] == BASE_EXP].copy()
test_df   = attach_weather(test_base)

###########################################
# Train XGBoost ensemble (2017)
# Train: Jan–Oct | Val: Nov–Dec
###########################################
val_mask = train_df["datetime"] >= "2017-11-01"
X_train  = train_df.loc[~val_mask, ENSEMBLE_FEATURES]
y_train  = train_df.loc[~val_mask, "residual"]
X_val    = train_df.loc[val_mask,  ENSEMBLE_FEATURES]
y_val    = train_df.loc[val_mask,  "residual"]
X_test   = test_df[ENSEMBLE_FEATURES]

print(f"Ensemble features : {ENSEMBLE_FEATURES}")
print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows | Test: {len(X_test):,} rows\n")

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
print(f"\nBest iteration: {model.best_iteration}")

test_df["predicted_corrected"] = test_df["predicted"] + model.predict(X_test)

###########################################
# Metrics
###########################################
def metrics(actual, pred, naive):
    mae   = (actual - pred).abs().mean()
    mape  = ((actual - pred).abs() / actual).mean() * 100
    smae  = mae / (actual - naive).abs().mean()
    return mae, mape, smae

print("\n=== Metrics (2018 test set) ===")

# Naive (use any experiment — actual/naive are the same across experiments)
df_ref = wf_2018[wf_2018["experiment"] == BASE_EXP]
mae_naive  = (df_ref["actual"] - df_ref["naive"]).abs().mean()
mape_naive = ((df_ref["actual"] - df_ref["naive"]).abs() / df_ref["actual"]).mean() * 100
print(f"  {'naive_lag24h':14s}: MAE={mae_naive:,.0f} MW | MAPE={mape_naive:.2f}%")
print(f"  {'-'*60}")

for exp_name, _ in EXPERIMENTS:
    df = wf_2018[wf_2018["experiment"] == exp_name]
    mae, mape, smae = metrics(df["actual"], df["predicted"], df["naive"])
    print(f"  {exp_name:14s}: MAE={mae:,.0f} MW | MAPE={mape:.2f}% | Scaled={smae:.3f}")

mae_ens, mape_ens, smae_ens = metrics(
    test_df["actual"], test_df["predicted_corrected"], test_df["naive"]
)
ensemble_label = f"ensemble (exp3 + XGBoost)"
print(f"  {ensemble_label}: MAE={mae_ens:,.0f} MW | MAPE={mape_ens:.2f}% | Scaled={smae_ens:.3f}")

###########################################
# Chart 1 — Full period: exp3 vs ensemble
###########################################
df_exp3 = wf_2018[wf_2018["experiment"] == BASE_EXP].copy()
df_exp3["datetime"] = df_exp3["date"] + pd.to_timedelta(df_exp3["hour"], unit="h")
df_exp3 = df_exp3.sort_values("datetime")

ens = test_df.sort_values("datetime")

fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
fig.suptitle("TimesFM vs Ensemble — Full Test Period (2018)", fontsize=18, y=1.01)

panels = [
    (df_exp3, "naive",                 ORANGE, "Naive (lag-24h)"),
    (df_exp3, "predicted",             BLUE,   "TimesFM (exp3_8736h)"),
    (ens,     "predicted_corrected",   GREEN,  "Ensemble (exp3 + XGBoost)"),
]

for ax, (src, pred_col, color, label) in zip(axes, panels):
    ax.plot(src["datetime"], src["actual"],   color=GRAY,  linewidth=0.6, alpha=0.8, label="Actual")
    ax.plot(src["datetime"], src[pred_col],   color=color, linewidth=0.6, alpha=0.9, label=label)
    ax.set_title(label, fontsize=16)
    ax.set_ylabel("MW", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_full_period.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Chart 2 — Zoom: 2-week window
###########################################
# Find the 14-day window where TimesFM vs Ensemble divergence is highest
daily_diff = (
    test_df.copy()
    .assign(abs_diff=lambda d: (d["predicted"] - d["predicted_corrected"]).abs())
    .groupby("date")["abs_diff"]
    .mean()
    .reset_index()
)
daily_diff["rolling_mean"] = daily_diff["abs_diff"].rolling(14).mean()
best_end   = daily_diff.loc[daily_diff["rolling_mean"].idxmax(), "date"]
best_start = best_end - pd.Timedelta(days=13)
zoom_start = pd.Timestamp(best_start)
zoom_end   = pd.Timestamp(best_end) + pd.Timedelta(hours=23)
print(f"\nBest zoom window: {zoom_start.date()} → {zoom_end.date()}")

fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
fig.suptitle("Zoom: 2-week window (Mar 2018)", fontsize=18, y=1.01)

# Panel 1 — Naive
df_naive = df_exp3.query("@zoom_start <= datetime <= @zoom_end")
axes[0].plot(df_naive["datetime"], df_naive["actual"], color=GRAY,   linewidth=1.5, alpha=0.8, label="Actual")
axes[0].plot(df_naive["datetime"], df_naive["naive"],  color=ORANGE, linewidth=1.5, alpha=0.9, label="Naive (lag-24h)")
axes[0].set_title("naive_lag24h", fontsize=16)

# Panel 2 — TimesFM exp3
df_zoom3 = df_exp3.query("@zoom_start <= datetime <= @zoom_end")
axes[1].plot(df_zoom3["datetime"], df_zoom3["actual"],    color=GRAY, linewidth=1.5, alpha=0.8, label="Actual")
axes[1].plot(df_zoom3["datetime"], df_zoom3["predicted"], color=BLUE, linewidth=1.5, alpha=0.9, label="TimesFM (exp3_8736h)")
axes[1].set_title("exp3_8736h", fontsize=16)

# Panel 3 — Ensemble
df_ens_zoom = ens.query("@zoom_start <= datetime <= @zoom_end")
axes[2].plot(df_ens_zoom["datetime"], df_ens_zoom["actual"],              color=GRAY,  linewidth=1.5, alpha=0.8, label="Actual")
axes[2].plot(df_ens_zoom["datetime"], df_ens_zoom["predicted_corrected"], color=GREEN, linewidth=1.5, alpha=0.9, label="Ensemble")
axes[2].set_title("Ensemble (exp3 + XGBoost)", fontsize=16)

for ax in axes:
    ax.set_ylabel("MW", fontsize=15)
    ax.tick_params(labelsize=14)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "ensemble_zoom.png", dpi=150, bbox_inches="tight")
plt.show()
