from pathlib import Path
import pandas as pd
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.timesfm_2p5.timesfm_2p5_base import ForecastConfig

DATA_PATH    = Path(__file__).parents[1] / "data/processed/pjme_clean.csv"
RESULTS_PATH = Path(__file__).parents[1] / "data/processed/walk_forward_results.csv"

HORIZON    = 24
TEST_START = pd.Timestamp("2018-01-01")
TEST_END   = pd.Timestamp("2018-07-31")

EXPERIMENTS = [
    ("exp1_512h",  512),
    ("exp2_672h",  672),
    ("exp3_8736h", 8736),
]

###########################################
# Load cleaned series
###########################################
series = (
    pd.read_csv(DATA_PATH, parse_dates=["Datetime"])
    .set_index("Datetime")
    .sort_index()
)

# Fix DST gaps (spring-forward = missing hours) and duplicates (fall-back = repeated hours)
series = series[~series.index.duplicated(keep="first")]
full_idx = pd.date_range(series.index.min(), series.index.max(), freq="h")
series = series.reindex(full_idx).interpolate(method="time")

###########################################
# Load model once (weights downloaded on first run)
# torch_compile=False — required for CPU
###########################################
tfm = TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
    torch_compile=False,
)

###########################################
# Walk-forward inference
###########################################
test_dates = pd.date_range(TEST_START, TEST_END, freq="D")
records = []

for exp_name, ctx_len in EXPERIMENTS:
    print(f"\n[{exp_name}] context={ctx_len}h | steps={len(test_dates)}")

    # Compile once per experiment with the exact context size
    # max_horizon is rounded up to next multiple of 128 (output patch size)
    tfm.compile(ForecastConfig(
        max_context=ctx_len,
        max_horizon=128,
        per_core_batch_size=1,
    ))

    for i, date in enumerate(test_dates):
        ctx = series.loc[
            date - pd.Timedelta(hours=ctx_len) : date - pd.Timedelta(hours=1),
            "mw",
        ].values

        actual = series.loc[
            date : date + pd.Timedelta(hours=HORIZON - 1),
            "mw",
        ].values

        if len(ctx) < ctx_len or len(actual) < HORIZON:
            continue

        point_forecast, _ = tfm.forecast(horizon=HORIZON, inputs=[ctx])
        pred  = point_forecast[0]
        naive = ctx[-HORIZON:]  # same hour from previous day (lag-24h seasonal naive)

        for h in range(HORIZON):
            records.append({
                "experiment":  exp_name,
                "context_len": ctx_len,
                "date":        date,
                "hour":        h,
                "actual":      actual[h],
                "predicted":   pred[h],
                "naive":       naive[h],
            })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(test_dates)} done")

###########################################
# Save
###########################################
results = pd.DataFrame(records)
results.to_csv(RESULTS_PATH, index=False)
print(f"\nSaved → {RESULTS_PATH}")
print(f"Rows: {len(results):,} ({results['experiment'].nunique()} experiments × {results['date'].nunique()} days × {HORIZON}h)")
