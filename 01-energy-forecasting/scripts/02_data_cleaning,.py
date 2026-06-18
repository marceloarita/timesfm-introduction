###########################################
# Import libraries
###########################################
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted")
BLUE = "#147EC5"
GRAY = "#95A5A6"
RED  = "#E74C3C"

DATA_PATH   = Path(__file__).parents[1] / "data/raw/pjm_hourly_est.csv"
OUT_PATH    = Path(__file__).parents[1] / "data/processed/pjme_clean.csv"
CHARTS_PATH = Path(__file__).parents[1] / "charts"
CHARTS_PATH.mkdir(exist_ok=True)

LOOKBACK_YEARS = 3
ZSCORE_THRESHOLD = 3

###########################################
# Load, filter & build single series
###########################################
df = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
df = df.sort_values("Datetime").reset_index(drop=True)

series = (
    df[["Datetime", "PJME"]]
    .dropna()
    .rename(columns={"PJME": "mw"})
    .query("'2002-01-01' <= Datetime < '2018-08-01'")
    .reset_index(drop=True)
)
print(f"Range  : {series['Datetime'].min()} → {series['Datetime'].max()}")
print(f"Points : {len(series):,}")

###########################################
# Helper columns
###########################################
series["month"]     = series["Datetime"].dt.month
series["hour"]      = series["Datetime"].dt.hour
series["dayofweek"] = series["Datetime"].dt.dayofweek

###########################################
# Sandy: detect recovery window via z-score
###########################################
SANDY_START = pd.Timestamp("2012-10-29 00:00")
SCAN_END    = pd.Timestamp("2012-11-15 00:00")  # max scan horizon

cutoff = SANDY_START - pd.DateOffset(years=LOOKBACK_YEARS)
reference = series[(series["Datetime"] >= cutoff) & (series["Datetime"] < SANDY_START)]
ref_stats = reference.groupby(["month", "dayofweek", "hour"])["mw"].agg(["mean", "std"])

post = series[(series["Datetime"] >= SANDY_START) & (series["Datetime"] <= SCAN_END)].copy()
post["ref_mean"] = post.apply(lambda r: ref_stats.loc[(r["month"], r["dayofweek"], r["hour"]), "mean"]
                              if (r["month"], r["dayofweek"], r["hour"]) in ref_stats.index else np.nan, axis=1)
post["ref_std"]  = post.apply(lambda r: ref_stats.loc[(r["month"], r["dayofweek"], r["hour"]), "std"]
                              if (r["month"], r["dayofweek"], r["hour"]) in ref_stats.index else np.nan, axis=1)
post["zscore_event"] = (post["mw"] - post["ref_mean"]) / post["ref_std"]

###########################################
# Plot: z-score over time after Sandy start
###########################################
fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
fig.subplots_adjust(hspace=0.4)

axes[0].plot(post["Datetime"], post["mw"] / 1000, color=BLUE, linewidth=1.2)
axes[0].set_title("PJM East Load — Post-Sandy Raw (Oct 29 – Nov 15, 2012)", fontsize=13, pad=12)
axes[0].set_ylabel("GW", fontsize=11)
axes[0].tick_params(labelsize=10)
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

axes[1].plot(post["Datetime"], post["zscore_event"], color=BLUE, linewidth=1.0)
axes[1].axhline( ZSCORE_THRESHOLD, color=RED,  linestyle="--", linewidth=1, label=f"+{ZSCORE_THRESHOLD}σ")
axes[1].axhline(-ZSCORE_THRESHOLD, color=RED,  linestyle="--", linewidth=1, label=f"-{ZSCORE_THRESHOLD}σ")
axes[1].axhline(0, color=GRAY, linestyle=":", linewidth=0.8)
axes[1].fill_between(post["Datetime"], post["zscore_event"],
                     where=post["zscore_event"].abs() > ZSCORE_THRESHOLD,
                     color=RED, alpha=0.25, label="Anomalous hours")
axes[1].set_title("Hourly z-score vs same day/hour in prior 3 years", fontsize=13, pad=12)
axes[1].set_ylabel("z-score", fontsize=11)
axes[1].tick_params(labelsize=10)
axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
axes[1].legend(loc="lower right", fontsize=10, framealpha=0.9)

sns.despine()
plt.savefig(CHARTS_PATH / "sandy_zscore_diagnostic.png", dpi=150, bbox_inches="tight")
plt.show()

print("Review the plot and set KNOWN_EVENTS end date below accordingly.")
print(f"Anomalous hours (|z|>{ZSCORE_THRESHOLD}): {(post['zscore_event'].abs() > ZSCORE_THRESHOLD).sum()}")
print(post[post["zscore_event"].abs() > ZSCORE_THRESHOLD][["Datetime", "mw", "zscore_event"]].tail(5).to_string())

###########################################
# Fill known events with pre-event historical mean
###########################################
# Set end date based on the z-score plot above
KNOWN_EVENTS = [
    ("2012-10-29 00:00", "2012-11-01 23:00"),  # Sandy: anomalous z-score clears ~Nov 1
]

for start, end in KNOWN_EVENTS:
    mask = (series["Datetime"] >= start) & (series["Datetime"] <= end)
    cutoff = pd.Timestamp(start) - pd.DateOffset(years=LOOKBACK_YEARS)
    pre_event = series[(series["Datetime"] >= cutoff) & (series["Datetime"] < start)]
    hist_mean = pre_event.groupby(["month", "dayofweek", "hour"])["mw"].mean()
    series.loc[mask, "mw"] = series.loc[mask].apply(
        lambda r: hist_mean.get((r["month"], r["dayofweek"], r["hour"]), np.nan), axis=1
    )
    print(f"Known event filled (hist mean, last {LOOKBACK_YEARS}y): {start} → {end}  ({mask.sum()} pts)")

###########################################
# Statistical outliers: seasonal z-score (month × hour)
###########################################
grp = series.groupby(["month", "hour"])["mw"]
series["zscore"] = grp.transform(lambda x: stats.zscore(x, ddof=1, nan_policy="omit"))

stat_outliers = series["zscore"].abs() > ZSCORE_THRESHOLD
series.loc[stat_outliers, "mw"] = np.nan
print(f"Statistical outliers masked: {stat_outliers.sum()} pts")

###########################################
# Linear interpolation for isolated statistical outliers
###########################################
series["mw"] = series["mw"].interpolate(method="linear", limit_direction="both")
print(f"NaN after interpolation: {series['mw'].isna().sum()}")

###########################################
# Validation plot: Sandy window
###########################################
raw = pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
raw = raw.sort_values("Datetime").rename(columns={"PJME": "mw_raw"})

window_start, window_end = "2012-10-25", "2012-11-10"
raw_w   = raw.set_index("Datetime")["mw_raw"].loc[window_start:window_end]
clean_w = series.set_index("Datetime")["mw"].loc[window_start:window_end]

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(raw_w.index,   raw_w.values,   color=GRAY, linewidth=1.2, label="Raw",     alpha=0.8)
ax.plot(clean_w.index, clean_w.values, color=BLUE, linewidth=1.2, label="Cleaned", alpha=0.9)
event_end = pd.Timestamp(KNOWN_EVENTS[0][1])
ax.axvspan(pd.Timestamp(KNOWN_EVENTS[0][0]), event_end,
           color=RED, alpha=0.12, label="Treated window")
ax.set(title="Data Cleaning — Sandy Window (Oct–Nov 2012)", ylabel="MW", xlabel="")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
sns.despine()
plt.tight_layout()
plt.show()

###########################################
# Save cleaned series
###########################################
out = series[["Datetime", "mw"]].copy()
out.to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
print(out.describe())
