"""
04_forecast_2025.py
────────────────────
Zero-shot validation: TimesFM vs Naive across 2023, 2024, and 2025.

Key question: does TimesFM consistently outperform the naive lag-52 baseline
across three consecutive years, without any fine-tuning on gasoline data?

Reuses existing caches where available (residual_walk_forward.csv for 2023,
baseline_results_2024.csv for 2024, forecast_2025_results.csv for 2025).

Outputs:
  charts/forecast_multiyear.png
  data/processed/forecast_2025_results.csv  (2025 cache)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from utils import eval_table, mae

sns.set_theme(style="white", palette="muted")

COLOR_ACTUAL = "#4B5563"   # soft charcoal — readable but not harsh
COLOR_TFM    = "#147EC5"   # project blue
COLOR_NAIVE  = "#B0BEC5"   # light blue-gray — clearly subordinate

BASE_DIR         = Path(__file__).parents[1]
DATA_PATH        = BASE_DIR / "data/raw/gasoline_weekly.csv"
RESIDUAL_CACHE   = BASE_DIR / "data/processed/residual_walk_forward.csv"
BASELINE_PATH    = BASE_DIR / "data/processed/baseline_results_2024.csv"
FORECAST_CACHE   = BASE_DIR / "data/processed/forecast_2025_results.csv"
CHARTS_PATH      = BASE_DIR / "charts"

CONTEXT_LEN   = 156
HORIZON       = 1
FORECAST_YEAR = 2025

# ── 1. Load demand ─────────────────────────────────────────────────────────────
demand = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
demand["naive"] = demand["kbpd"].shift(52)
naive_lookup    = demand.set_index("date")["naive"]

values = demand["kbpd"].to_numpy(dtype=float)
dates  = demand["date"]
print(f"Demand: {demand['date'].min().date()} -> {demand['date'].max().date()}")

# ── 2. Load 2023 from residual cache ──────────────────────────────────────────
if not RESIDUAL_CACHE.exists():
    raise FileNotFoundError(f"{RESIDUAL_CACHE} not found.\nRun 03_ensemble.py first.")

residual_df = pd.read_csv(RESIDUAL_CACHE, parse_dates=["date"])
df_2023 = (
    residual_df[residual_df["date"].dt.year == 2023]
    [["date", "actual", "timesfm_pred"]]
    .rename(columns={"timesfm_pred": "timesfm"})
    .copy()
)
df_2023["naive"] = df_2023["date"].map(naive_lookup)
df_2023 = df_2023.dropna().reset_index(drop=True)
print(f"2023: {len(df_2023)} weeks from residual cache")

# ── 3. Load 2024 from baseline cache ──────────────────────────────────────────
if not BASELINE_PATH.exists():
    raise FileNotFoundError(f"{BASELINE_PATH} not found.\nRun 02_baseline_models.py first.")

df_2024 = (
    pd.read_csv(BASELINE_PATH, parse_dates=["date"])
    [["date", "actual", "naive", "timesfm"]]
    .copy()
)
print(f"2024: {len(df_2024)} weeks from baseline cache")

# ── 4. Load or run 2025 ───────────────────────────────────────────────────────
if FORECAST_CACHE.exists():
    df_2025 = pd.read_csv(FORECAST_CACHE, parse_dates=["date"])
    print(f"2025: {len(df_2025)} weeks from forecast cache")
else:
    n_2025 = (demand["date"].dt.year == FORECAST_YEAR).sum()
    if n_2025 == 0:
        raise ValueError(f"No {FORECAST_YEAR} data. Re-run 00_download.py.")
    print(f"\nRunning TimesFM walk-forward on {FORECAST_YEAR}...")

    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
    from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", torch_compile=False
    )
    tfm.compile(ForecastConfig(max_context=CONTEXT_LEN, max_horizon=128, per_core_batch_size=1))

    positions_2025 = demand[demand["date"].dt.year == FORECAST_YEAR].index.tolist()
    records = []
    for i, pos in enumerate(positions_2025):
        if pos < CONTEXT_LEN + 52:
            continue
        ctx   = values[pos - CONTEXT_LEN : pos]
        pf, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
        records.append({
            "date":    dates[pos],
            "actual":  values[pos],
            "naive":   values[pos - 52],
            "timesfm": float(pf[0][0]),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(positions_2025)} done")

    df_2025 = pd.DataFrame(records).reset_index(drop=True)
    FORECAST_CACHE.parent.mkdir(exist_ok=True)
    df_2025.to_csv(FORECAST_CACHE, index=False)
    print(f"  Saved -> {FORECAST_CACHE}")

# ── 5. Combine and evaluate ────────────────────────────────────────────────────
all_years = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
all_years["year"] = all_years["date"].dt.year

print()
rows = []
for year, grp in all_years.groupby("year"):
    rows.append((f"Naive lag-52  ({year})", grp["naive"].values,
                 grp["actual"].values, grp["naive"].values))
    rows.append((f"TimesFM       ({year})", grp["timesfm"].values,
                 grp["actual"].values, grp["naive"].values))
eval_table(rows, title="Zero-Shot Validation — 2023 / 2024 / 2025")

# ── 6. Chart — 3-year forecast vs actual ──────────────────────────────────────
from matplotlib.transforms import blended_transform_factory

year_boundaries = [
    all_years[all_years["year"] == yr]["date"].max()
    for yr in [2023, 2024]
]

# Pre-compute per-year Scaled MAE
year_stats = {}
for year, grp in all_years.groupby("year"):
    smae = mae(grp["timesfm"].values, grp["actual"].values) / mae(grp["naive"].values, grp["actual"].values)
    mid  = grp["date"].iloc[len(grp) // 2]
    year_stats[year] = {"smae": smae, "mid": mid}

fig, ax = plt.subplots(figsize=(16, 5))

ax.plot(all_years["date"], all_years["actual"],  color=COLOR_ACTUAL, linewidth=1.6,
        label="Actual demand", zorder=4)
ax.plot(all_years["date"], all_years["naive"],   color=COLOR_NAIVE,  linewidth=1.2,
        linestyle="--", label="Naive lag-52", zorder=2)
ax.plot(all_years["date"], all_years["timesfm"], color=COLOR_TFM,    linewidth=1.8,
        label="TimesFM exp_156w (zero-shot)", zorder=3)

# Light alternating background per year
year_colors = ["#F8FAFC", "#FFFFFF", "#F8FAFC"]
for yr, bg in zip([2023, 2024, 2025], year_colors):
    grp = all_years[all_years["year"] == yr]
    ax.axvspan(grp["date"].min(), grp["date"].max(), color=bg, alpha=1.0, zorder=0)

# Year separator lines
for boundary in year_boundaries:
    ax.axvline(boundary, color="#CBD5E0", linewidth=1.0, linestyle="--", zorder=1)

# Annotations: blended transform keeps x in data coords, y in axes fraction
# Placed at top of axes (0.97) to avoid clashing with data, below title
trans = blended_transform_factory(ax.transData, ax.transAxes)
for year, stats in year_stats.items():
    ax.text(
        stats["mid"], 0.96,
        f"{year}   Scaled MAE = {stats['smae']:.3f}",
        transform=trans, ha="center", va="top",
        fontsize=11, color="#6B7280",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#E2E8F0", alpha=0.92),
    )

ax.set_title("U.S. Gasoline Demand — TimesFM Zero-Shot vs Naive (2023–2025)",
             fontsize=14, pad=14)
ax.set_ylabel("Thousand Barrels/Day", fontsize=12)
ax.tick_params(labelsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.1f}k"))
ax.legend(fontsize=11, framealpha=0.95, loc="lower left")
sns.despine()
plt.tight_layout()
plt.savefig(CHARTS_PATH / "forecast_multiyear.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nChart saved to {CHARTS_PATH / 'forecast_multiyear.png'}")
