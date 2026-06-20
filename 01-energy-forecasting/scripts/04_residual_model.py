from pathlib import Path
import pandas as pd
import requests
import xgboost as xgb
import shap
import holidays as hd
import matplotlib.pyplot as plt
import seaborn as sns
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

sns.set_theme(style="white", palette="muted")

DATA_PATH        = Path(__file__).parents[1] / "data/processed/pjme_clean.csv"
WF_2018_PATH     = Path(__file__).parents[1] / "data/processed/walk_forward_results.csv"
WF_2017_PATH     = Path(__file__).parents[1] / "data/processed/walk_forward_2017.csv"
WEATHER_PATH     = Path(__file__).parents[1] / "data/processed/weather_philadelphia.csv"
XGB_RESULTS_PATH = Path(__file__).parents[1] / "data/processed/xgb_residual_results.csv"
CHARTS_PATH      = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

BASE_EXP    = "exp3_8736h"
BASE_CTX    = 8736
HORIZON     = 24
TRAIN_START = pd.Timestamp("2017-01-01")
TRAIN_END   = pd.Timestamp("2017-12-31")

# Philadelphia — representative city for PJM East region
LAT, LON = 39.95, -75.16

# All candidate features for SHAP discovery
FEATURES_ALL = [
    "hour_of_day", "dayofweek", "month", "is_weekend", "is_holiday",
    "temperature", "apparent_temperature", "humidity", "HDD", "CDD",
]

# Weather-only features selected after SHAP analysis
FEATURES_SELECTED = ["temperature", "apparent_temperature", "humidity", "HDD", "CDD"]

###########################################
# Load cleaned series (with DST fix)
###########################################
series = (
    pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    .set_index("Datetime")
    .sort_index()
)
series = series[~series.index.duplicated(keep="first")]
full_idx = pd.date_range(series.index.min(), series.index.max(), freq="h")
series = series.reindex(full_idx).interpolate(method="time")

###########################################
# 2017 walk-forward (cached)
###########################################
if WF_2017_PATH.exists():
    print(f"Loading cached 2017 walk-forward → {WF_2017_PATH}")
    wf_2017 = pd.read_csv(WF_2017_PATH, parse_dates=["date"])
else:
    print(f"Running 2017 walk-forward ({BASE_EXP})...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False,
    )
    tfm.compile(ForecastConfig(
        max_context=BASE_CTX,
        max_horizon=128,
        per_core_batch_size=1,
    ))

    train_dates = pd.date_range(TRAIN_START, TRAIN_END, freq="D")
    records = []

    for i, date in enumerate(train_dates):
        ctx = series.loc[
            date - pd.Timedelta(hours=BASE_CTX) : date - pd.Timedelta(hours=1),
            "mw",
        ].values
        actual = series.loc[
            date : date + pd.Timedelta(hours=HORIZON - 1),
            "mw",
        ].values

        if len(ctx) < BASE_CTX or len(actual) < HORIZON:
            continue

        point_forecast, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
        pred  = point_forecast[0]
        naive = ctx[-HORIZON:]

        for h in range(HORIZON):
            records.append({
                "date":      date,
                "hour":      h,
                "actual":    actual[h],
                "predicted": pred[h],
                "naive":     naive[h],
            })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(train_dates)} done")

    wf_2017 = pd.DataFrame(records)
    wf_2017.to_csv(WF_2017_PATH, index=False)
    print(f"Saved → {WF_2017_PATH}")

###########################################
# Weather data — Open-Meteo (cached)
###########################################
if WEATHER_PATH.exists():
    print(f"Loading cached weather → {WEATHER_PATH}")
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"]).set_index("datetime")
else:
    print("Fetching weather from Open-Meteo (Philadelphia, 2017–2018)...")
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date=2017-01-01&end_date=2018-07-31"
        f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m"
        f"&timezone=America%2FNew_York"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    weather = pd.DataFrame({
        "datetime":             pd.to_datetime(data["time"]),
        "temperature":          data["temperature_2m"],
        "apparent_temperature": data["apparent_temperature"],
        "humidity":             data["relative_humidity_2m"],
    }).set_index("datetime")
    weather.to_csv(WEATHER_PATH)
    print(f"Saved → {WEATHER_PATH}")

###########################################
# Feature engineering — all candidate features
###########################################
us_holidays = hd.US(years=[2017, 2018])

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["datetime"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")
    df["residual"]  = df["actual"] - df["predicted"]

    dt = df["datetime"]
    df["hour_of_day"] = dt.dt.hour
    df["dayofweek"]   = dt.dt.dayofweek
    df["month"]       = dt.dt.month
    df["is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    df["is_holiday"]  = dt.dt.date.apply(lambda d: int(d in us_holidays))

    df = df.join(weather, on="datetime")
    df["HDD"] = (18 - df["temperature"]).clip(lower=0)
    df["CDD"] = (df["temperature"] - 18).clip(lower=0)

    return df.dropna(subset=FEATURES_ALL + ["residual"])

train_df = build_features(wf_2017)

wf_2018 = pd.read_csv(WF_2018_PATH, parse_dates=["date"])
wf_2018 = wf_2018[wf_2018["experiment"] == BASE_EXP].copy()
test_df  = build_features(wf_2018)

val_mask = train_df["datetime"] >= "2017-11-01"
y_train  = train_df.loc[~val_mask, "residual"]
y_val    = train_df.loc[val_mask,  "residual"]

print(f"\nTrain: {(~val_mask).sum():,} rows | Val: {val_mask.sum():,} rows | Test: {len(test_df):,} rows")

###########################################
# Phase 1 — Train with ALL features (SHAP discovery)
###########################################
X_train_all = train_df.loc[~val_mask, FEATURES_ALL]
X_val_all   = train_df.loc[val_mask,  FEATURES_ALL]

xgb_full = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
xgb_full.fit(X_train_all, y_train, eval_set=[(X_val_all, y_val)], verbose=50)
print(f"\nBest iteration (full model): {xgb_full.best_iteration}")

###########################################
# SHAP feature importance — all features
###########################################
explainer_full  = shap.TreeExplainer(xgb_full)
shap_sample_all = X_train_all.sample(min(2000, len(X_train_all)), random_state=42)
shap_values_all = explainer_full.shap_values(shap_sample_all)

shap.summary_plot(shap_values_all, shap_sample_all, show=False, plot_size=(14, 5))
plt.title("SHAP Feature Importance — XGBoost Residual Model (all features)", fontsize=14, pad=10)
plt.tick_params(labelsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "shap_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Phase 2 — Retrain with SHAP-selected features (weather only)
###########################################
print("\n→ SHAP shows weather features dominate. Retraining with selected features...")

X_train_sel = train_df.loc[~val_mask, FEATURES_SELECTED]
X_val_sel   = train_df.loc[val_mask,  FEATURES_SELECTED]
X_test_sel  = test_df[FEATURES_SELECTED]

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train_sel, y_train, eval_set=[(X_val_sel, y_val)], verbose=50)
print(f"\nBest iteration (selected): {xgb_model.best_iteration}")

###########################################
# Predict 2018 residuals & save
###########################################
test_df["xgb_residual_pred"]   = xgb_model.predict(X_test_sel)
test_df["predicted_corrected"] = test_df["predicted"] + test_df["xgb_residual_pred"]

out_cols = ["date", "hour", "actual", "predicted", "xgb_residual_pred", "predicted_corrected", "naive"]
test_df[out_cols].to_csv(XGB_RESULTS_PATH, index=False)
print(f"\nSaved → {XGB_RESULTS_PATH}")

###########################################
# Quick metrics preview
###########################################
mae_naive    = (test_df["actual"] - test_df["naive"]).abs().mean()
mae_timesfm  = (test_df["actual"] - test_df["predicted"]).abs().mean()
mae_ensemble = (test_df["actual"] - test_df["predicted_corrected"]).abs().mean()

mape_naive    = ((test_df["actual"] - test_df["naive"]).abs() / test_df["actual"]).mean() * 100
mape_timesfm  = ((test_df["actual"] - test_df["predicted"]).abs() / test_df["actual"]).mean() * 100
mape_ensemble = ((test_df["actual"] - test_df["predicted_corrected"]).abs() / test_df["actual"]).mean() * 100

print("\n=== Quick metrics (2018 test set) ===")
print(f"  naive_lag24h : MAE={mae_naive:,.0f} MW | MAPE={mape_naive:.2f}%")
print(f"  TimesFM only : MAE={mae_timesfm:,.0f} MW | MAPE={mape_timesfm:.2f}% | Scaled={mae_timesfm/mae_naive:.3f}")
print(f"  Ensemble (XGBoost + TimesFM)     : MAE={mae_ensemble:,.0f} MW | MAPE={mape_ensemble:.2f}% | Scaled={mae_ensemble/mae_naive:.3f}")

###########################################
# SHAP analysis — temperature & overestimation
# Hypothesis: TimesFM overestimates high-temperature days because
# 2017 summer context (used for training) had higher consumption than 2018
###########################################
explainer_sel   = shap.TreeExplainer(xgb_model)
shap_sample_sel = X_train_sel.sample(min(2000, len(X_train_sel)), random_state=42)
shap_values_sel = explainer_sel.shap_values(shap_sample_sel)

temp_idx  = FEATURES_SELECTED.index("temperature")
shap_temp = shap_values_sel[:, temp_idx]
temp_vals = shap_sample_sel["temperature"].values

# 2017 vs 2018 summer hourly mean (Jun–Jul overlap)
summer_2017      = series["2017-06-01":"2017-07-31"].copy()
summer_2017_mean = summer_2017.groupby(summer_2017.index.hour)["mw"].mean()

summer_2018      = series["2018-06-01":"2018-07-31"].copy()
summer_2018_mean = summer_2018.groupby(summer_2018.index.hour)["mw"].mean()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Temperature & Overestimation Analysis", fontsize=18, y=1.02)

# Panel 1 — Residual vs Temperature (Jun–Jul 2018 only)
ax = axes[0]
summer_test = test_df[test_df["month"].isin([6, 7])]
ax.scatter(summer_test["temperature"], summer_test["residual"],
           color="#E74C3C", alpha=0.3, s=12, rasterized=True)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Temperature (°C)", fontsize=15)
ax.set_ylabel("Residual — actual minus TimesFM (MW)", fontsize=13)
ax.set_title("Residual vs Temperature\n(Jun–Jul 2018)", fontsize=16)
ax.tick_params(labelsize=14)
sns.despine(ax=ax)

# Panel 2 — SHAP dependence plot for temperature (selected model, training data)
ax = axes[1]
sc = ax.scatter(temp_vals, shap_temp, c=temp_vals, cmap="coolwarm", alpha=0.4, s=10, rasterized=True)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
plt.colorbar(sc, ax=ax, label="Temperature (°C)")
ax.set_xlabel("Temperature (°C)", fontsize=15)
ax.set_ylabel("SHAP value (residual correction, MW)", fontsize=13)
ax.set_title("SHAP Dependence — Temperature\n(2017 training set)", fontsize=16)
ax.tick_params(labelsize=14)
sns.despine(ax=ax)

# Panel 3 — 2017 vs 2018 summer hourly mean (hypothesis validation)
ax = axes[2]
hours = summer_2017_mean.index
ax.plot(hours, summer_2017_mean.values / 1000, color="#E74C3C", linewidth=2.5, marker="o", markersize=5, label="2017 (train context)")
ax.plot(hours, summer_2018_mean.values / 1000, color="#3498DB", linewidth=2.5, marker="o", markersize=5, label="2018 (test)")
ax.fill_between(hours,
                summer_2017_mean.values / 1000,
                summer_2018_mean.values / 1000,
                alpha=0.15, color="gray", label="Difference")
ax.set_xlabel("Hour of day", fontsize=15)
ax.set_ylabel("Mean consumption (GW)", fontsize=15)
ax.set_title("Summer load profile: 2017 vs 2018\n(Jun–Jul, hypothesis validation)", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(fontsize=12, framealpha=0.9)
sns.despine(ax=ax)

diff_pct = ((summer_2017_mean - summer_2018_mean) / summer_2018_mean * 100).mean()
print(f"\nSummer mean consumption — 2017 vs 2018:")
print(f"  2017 (Jun–Jul): {summer_2017_mean.mean()/1000:.2f} GW")
print(f"  2018 (Jun–Jul): {summer_2018_mean.mean()/1000:.2f} GW")
print(f"  Difference    : {diff_pct:+.1f}% (positive = 2017 higher than 2018)")

plt.tight_layout()
plt.savefig(CHARTS_PATH / "shap_temperature_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

###########################################
# Chart — TimesFM residual vs Ensemble residual (temp > 25°C)
###########################################
hot = test_df[test_df["temperature"] > 25].copy()
hot["residual_ensemble"] = hot["actual"] - hot["predicted_corrected"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
fig.suptitle("Residual comparison — temperature > 25 °C (2018)", fontsize=18, y=1.02)

for ax, (col, label, color) in zip(axes, [
    ("residual",          "TimesFM (exp3_8736h)", "#147EC5"),
    ("residual_ensemble", "Ensemble (exp3 + XGBoost)", "#AF4DED"),
]):
    ax.scatter(hot["temperature"], hot[col],
               color=color, alpha=0.35, s=14, rasterized=True)
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    mean_res = hot[col].mean()
    ax.axhline(mean_res, color=color, linewidth=1.5, linestyle=":",
               label=f"Mean residual: {mean_res:+,.0f} MW")
    ax.set_xlabel("Temperature (°C)", fontsize=15)
    ax.set_ylabel("Residual — actual minus predicted (MW)", fontsize=13)
    ax.set_title(label, fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12, framealpha=0.9)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "residual_hot_days_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nHot days (temp > 25°C) — residual summary:")
print(f"  TimesFM  mean residual : {hot['residual'].mean():+,.0f} MW")
print(f"  Ensemble mean residual : {hot['residual_ensemble'].mean():+,.0f} MW")

###########################################
# XGBoost residual model quality (2018 test)
# Does XGBoost (weather-only) actually predict the residual well?
###########################################
ss_res   = ((test_df["residual"] - test_df["xgb_residual_pred"]) ** 2).sum()
ss_tot   = ((test_df["residual"] - test_df["residual"].mean()) ** 2).sum()
r2_xgb   = 1 - ss_res / ss_tot
r_corr   = test_df["residual"].corr(test_df["xgb_residual_pred"])
rmse_xgb = (ss_res / len(test_df)) ** 0.5
mae_xgb  = (test_df["residual"] - test_df["xgb_residual_pred"]).abs().mean()

print("\n=== XGBoost residual model quality (2018) ===")
print(f"  Pearson r              : {r_corr:.3f}")
print(f"  R²                     : {r2_xgb:.3f}")
print(f"  RMSE (residual error)  : {rmse_xgb:,.0f} MW")
print(f"  MAE  (residual error)  : {mae_xgb:,.0f} MW")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("XGBoost Residual Model Quality — 2018 Test Set", fontsize=18, y=1.02)

# Panel 1 — Scatter: XGBoost predicted residual vs actual residual
ax = axes[0]
lim = max(test_df["residual"].abs().max(), test_df["xgb_residual_pred"].abs().max()) * 1.05
ax.scatter(test_df["xgb_residual_pred"], test_df["residual"],
           alpha=0.15, s=8, color="#2980B9", rasterized=True)
ax.plot([-lim, lim], [-lim, lim], color="black", linewidth=1, linestyle="--", label="Identity line")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("XGBoost predicted residual (MW)", fontsize=13)
ax.set_ylabel("Actual residual — actual minus TimesFM (MW)", fontsize=13)
ax.set_title(f"Predicted vs Actual Residual\nR² = {r2_xgb:.3f}  |  r = {r_corr:.3f}", fontsize=15)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
sns.despine(ax=ax)

# Panel 2 — Time series overlay (Jan–Mar 2018)
ax = axes[1]
sample = (
    test_df[test_df["date"] <= pd.Timestamp("2018-03-31")]
    .set_index("datetime")
    .sort_index()
)
ax.plot(sample.index, sample["residual"],
        color="#E74C3C", linewidth=1, alpha=0.85, label="Actual residual")
ax.plot(sample.index, sample["xgb_residual_pred"],
        color="#27AE60", linewidth=1, alpha=0.85, label="XGBoost prediction")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Residual (MW)", fontsize=13)
ax.set_title("Residual: Actual vs XGBoost Predicted\n(Jan–Mar 2018)", fontsize=15)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11)
sns.despine(ax=ax)

# Panel 3 — Distribution of residual prediction error
ax = axes[2]
res_error = test_df["residual"] - test_df["xgb_residual_pred"]
sns.histplot(res_error, bins=60, color="#8E44AD", kde=True, ax=ax)
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.axvline(res_error.mean(), color="#E74C3C", linewidth=1.5, linestyle=":",
           label=f"Mean: {res_error.mean():+,.0f} MW")
ax.set_xlabel("Residual prediction error — actual minus XGBoost (MW)", fontsize=12)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Distribution of Residual Prediction Error", fontsize=15)
ax.tick_params(labelsize=12)
ax.legend(fontsize=11)
sns.despine(ax=ax)

plt.tight_layout()
plt.savefig(CHARTS_PATH / "xgb_residual_quality.png", dpi=150, bbox_inches="tight")
plt.show()
