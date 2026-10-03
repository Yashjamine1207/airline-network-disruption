#!/usr/bin/env python
"""Phase 7A, Step 10b: how much does a trained sequence network rely on its lookback? (validation only)

Save as:   scripts/run_phase7a_sequence_perturbation.py
Run from the project root, after the network was trained (scripts/train_phase6_sequence_model.py):

    # 1. Smoke run on a subset of rows (about a minute). Tagged "smoke"; numbers are not results.
    python scripts/run_phase7a_sequence_perturbation.py --arch lstm --max-rows 60000

    # 2. Real run (LSTM, about 10 minutes on 16 cores). Repeat with --arch gru if you want it.
    python scripts/run_phase7a_sequence_perturbation.py --arch lstm

What it does
------------
Loads the saved network (seed 42, default configuration), scores the SECOND half of 2022 once as trained, then
again after each change to the airport lookback:

    baseline               nothing changed. Must reproduce the stored predictions or the run stops.
    no_sequence            lookback removal: no real step at all (the network sees only its static inputs)
    shuffled_lookbacks     permutation: every flight receives another flight's lookback (from the same batch of
                           2,048 consecutive prediction times, so the time of day is realistic but the airport is not)
    keep_newest_1/4/8      shorter lookback: only the newest k bins are kept
    mask_oldest ... newest temporal masking: one quarter of the lookback is hidden at a time (performance by
                           sequence position)

For every change: PR-AUC, the change in PR-AUC against the baseline with a paired day-level bootstrap interval,
the mean absolute change in the score, and the rank correlation with the baseline scores.

It also reports the baseline by sequence availability (how many real bins the flight had).

What this does and does not show
--------------------------------
A large drop means THIS trained network uses that input. It is a statement about the network, not about
the airline system. A network shown an input it never met in training (for example an empty lookback for a
flight that has history) can behave in odd ways, so a drop is an upper bound on how much the input is
worth. No drop means the network ignores that part. Nothing here is a causal effect.

2023 is never loaded.

Writes (small tables, safe to commit):
    reports/tables/phase7a_sequence_perturbation_<arch>.csv
    reports/tables/phase7a_sequence_availability_<arch>.csv
    reports/tables/phase7a_sequence_perturbation_<arch>_decision.json
(a smoke run adds "_smoke" to the file names)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
try:
    from airline_disruption.deep import inputs as di
    from airline_disruption.deep.perturb import PerturbedSource, build_passes, real_steps, verdict
    from airline_disruption.deep import protocol as proto
    from airline_disruption.deep.batching import BatchSource
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap
    from airline_disruption.explain import errors
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.deep import inputs as di  # noqa: E402
    from airline_disruption.deep.perturb import PerturbedSource, build_passes, real_steps, verdict  # noqa: E402
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.deep.batching import BatchSource  # noqa: E402
    from airline_disruption.evaluation.classification import binary_metrics, paired_day_bootstrap  # noqa: E402
    from airline_disruption.explain import errors  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

INPUTS_DIR = "data/processed/phase6/deep_inputs_v1"
TABLE_PATH = "data/processed/phase6/airport_bins_v1.npz"
REPRODUCTION_TOLERANCE = 1e-4  # PR-AUC difference between the reloaded network and the stored scores
MIN_EVENTS = 30


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7A sequence perturbation study (validation only)")
    parser.add_argument("--arch", default="lstm", choices=("lstm", "gru", "transformer"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inputs", default=INPUTS_DIR)
    parser.add_argument("--model-file", default=None, help="default data/processed/phase6/models/<arch>_c0_s42.keras")
    parser.add_argument("--stored-predictions", default=None, help="default data/processed/phase6/predictions/<arch>_validation.parquet")
    parser.add_argument("--stored-column", default=None, help="default: the architecture name")
    parser.add_argument("--lookback", type=int, default=None, help="default: read from the saved model")
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=100, help="day-level bootstrap resamples")
    parser.add_argument("--max-rows", type=int, default=None, help="score a random subset of the rows, smoke runs only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    arch = args.arch

    smoke = args.max_rows is not None
    tag = args.tag or ("smoke" if smoke else "")
    infix = f"_{tag}" if tag else ""
    model_file = root / (args.model_file or f"data/processed/phase6/models/{arch}_c0_s42.keras")
    if args.model_file is None and not model_file.exists():
        # The training run saves one file per configuration and seed. Use the only seed-42 file if there is exactly one.
        found = sorted((root / "data/processed/phase6/models").glob(f"{arch}_c*_s42.keras"))
        if len(found) == 1:
            model_file = found[0]
        elif found:
            fail(f"several saved {arch} models for seed 42: {[f.name for f in found]}. Pass --model-file (the configuration chosen in the training run).")
    stored_file = root / (args.stored_predictions or f"data/processed/phase6/predictions/{arch}_validation.parquet")
    stored_column = args.stored_column or arch
    for path in (root / args.inputs, root / TABLE_PATH, model_file, stored_file):
        if not path.exists():
            fail(f"not found: {path}")
    try:
        from airline_disruption.deep import keras_models as km
    except ImportError as problem:
        fail(f"TensorFlow is needed for this script ({problem})")
    km.set_threads(args.n_threads)
    if smoke:
        print("SMOKE RUN: a subset of rows. Numbers are not results.")

    inputs = di.load_deep_inputs(root / args.inputs)
    if inputs.manifest.get("final_test_used") is not False:
        fail("the inputs manifest does not say the final test was excluded")
    table, _, grid = ab.load_bin_table(root / TABLE_PATH)
    val_rows = inputs.rows("validation")
    seconds = inputs.prediction_seconds
    stamps = pd.Series(pd.to_datetime(seconds[val_rows], unit="s", utc=True))
    try:
        _, confirmation = tune.validation_halves(stamps)
    except ValueError as problem:
        fail(str(problem))
    rows = val_rows[confirmation]
    rows = proto.subsample_rows(rows, args.max_rows, args.seed)
    if seconds[rows].max() >= 1_672_531_200:
        fail("a row from 2023 or later is in the scored rows")
    y = inputs.label[rows]
    ids = inputs.source_row_number[rows]
    days = pd.to_datetime(seconds[rows], unit="s", utc=True).floor("D").to_numpy()
    try:
        proto.require_two_classes(y, "scored")
        stored = proto.align_scores(ids, y, pd.read_parquet(stored_file), stored_column)
    except ValueError as problem:
        fail(str(problem))
    print(f"Scoring {len(rows):,} rows of the second half of 2022 ({int(y.sum()):,} events) with {model_file.name}")

    model = km.load_model(model_file)
    sequence_inputs = [tensor for tensor in model.inputs if getattr(tensor, "name", "").startswith("sequence") and "mask" not in tensor.name]
    model_lookback = int(sequence_inputs[0].shape[1]) if sequence_inputs else None
    lookback = args.lookback or model_lookback
    if lookback is None:
        fail("could not read the lookback length from the model; pass --lookback")
    if model_lookback is not None and lookback != model_lookback:
        fail(f"--lookback {lookback} differs from the saved model's {model_lookback}")
    print(f"  lookback: {lookback} bins")

    def score(change) -> np.ndarray:
        source = PerturbedSource(inputs, rows, table, grid, batch_size=2048, lookback=lookback, use_sequence=True, change=change)
        return km.predict_scores(model, source)

    try:
        passes = build_passes(lookback, args.seed)
    except ValueError as problem:
        fail(str(problem))
    scores: dict[str, np.ndarray] = {}
    for name, _, change in passes:
        t0 = time.perf_counter()
        scores[name] = score(change)
        print(f"  scored {name:<28} {time.perf_counter() - t0:5.0f} s", flush=True)
    km.release()

    baseline = scores["baseline"]
    pr_baseline = float(binary_metrics(y, baseline)["pr_auc"])
    pr_stored = float(binary_metrics(y, stored)["pr_auc"])
    reproduction = {"pr_auc_reloaded": pr_baseline, "pr_auc_stored": pr_stored, "pr_auc_difference": pr_baseline - pr_stored,
                    "max_abs_score_difference": float(np.abs(baseline - stored).max()), "tolerance": REPRODUCTION_TOLERANCE}
    print(f"\nReloaded network vs stored predictions: PR-AUC {pr_baseline:.6f} vs {pr_stored:.6f}, "
          f"largest score difference {reproduction['max_abs_score_difference']:.2e}")
    if abs(reproduction["pr_auc_difference"]) > REPRODUCTION_TOLERANCE:
        fail("the reloaded network does not reproduce the stored predictions. Wrong model file, inputs or lookback?")

    # ------------------------------------------------------------------
    # Table: one row per pass
    # ------------------------------------------------------------------
    order_baseline = pd.Series(baseline).rank()
    result_rows = []
    for name, description, _ in passes:
        s = scores[name]
        record = {"arch": arch, "perturbation": name, "description": description, "rows": int(len(y)), "events": int(y.sum()),
                  "pr_auc": float(binary_metrics(y, s)["pr_auc"]), "mean_score": float(s.mean()),
                  "mean_abs_score_change": float(np.abs(s - baseline).mean()),
                  "rank_correlation_with_baseline": float(pd.Series(s).rank().corr(order_baseline))}
        if name != "baseline" and args.bootstrap > 0:
            outcome = paired_day_bootstrap(y, baseline, s, days, n_boot=args.bootstrap, seed=args.seed)
            record.update({"delta_pr_auc": outcome["difference"], "ci_low": outcome["ci_low"], "ci_high": outcome["ci_high"],
                           "verdict": verdict(outcome["difference"], outcome["ci_low"], outcome["ci_high"])})
        elif name != "baseline":
            record.update({"delta_pr_auc": record["pr_auc"] - pr_baseline, "ci_low": np.nan, "ci_high": np.nan, "verdict": "bootstrap skipped"})
        result_rows.append(record)
    result = pd.DataFrame(result_rows)
    print("\nChange in PR-AUC after each change to the lookback (second half of 2022):")
    show = ["perturbation", "pr_auc", "delta_pr_auc", "ci_low", "ci_high", "mean_abs_score_change", "rank_correlation_with_baseline"]
    print(result[show].to_string(index=False, float_format=lambda v: f"{v:+.5f}" if abs(v) < 1 else f"{v:.4f}"))
    for _, row in result.dropna(subset=["verdict"]).iterrows():
        print(f"  {row['perturbation']}: {row['verdict']}")

    # ------------------------------------------------------------------
    # Baseline by sequence availability
    # ------------------------------------------------------------------
    real = real_steps(BatchSource(inputs, rows, table, grid, batch_size=2048, lookback=lookback, use_sequence=True))
    bands = pd.cut(real, [-1, 0, lookback // 2 - 1, lookback - 1, lookback],
                   labels=["no history (0 real bins)", f"1 to {lookback // 2 - 1} real bins", f"{lookback // 2} to {lookback - 1} real bins", f"full ({lookback} real bins)"])
    frame = pd.DataFrame({"label": y, "score": baseline, "probability": baseline, "group": bands.astype("object")})
    cutoffs = errors.top_fraction_cutoffs(baseline, (0.01, 0.10))
    availability = errors.group_metrics(frame, "sequence_availability", cutoffs, "group", min_events=MIN_EVENTS)
    print("\nBaseline by how much lookback the flight had:")
    if len(availability) < 2:
        print("  (every scored flight has the same amount of lookback, so sequence availability cannot be compared in 2022)")
    print(availability[["group", "rows", "events", "prevalence", "recall_top1pct", "recall_top10pct", "pr_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    tables = root / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    result.to_csv(tables / f"phase7a_sequence_perturbation_{arch}{infix}.csv", index=False)
    availability.to_csv(tables / f"phase7a_sequence_availability_{arch}{infix}.csv", index=False)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tag": tag, "smoke_run": smoke, "arch": arch,
        "model_file": str(model_file.name), "seed_of_saved_model": 42, "lookback_bins": lookback,
        "rows_scored": int(len(y)), "period": "2022 second half (validation)",
        "reproduction_check": reproduction, "bootstrap_resamples": args.bootstrap,
        "passes": result[["perturbation", "delta_pr_auc", "ci_low", "ci_high", "verdict"]].dropna(subset=["verdict"]).to_dict("records"),
        "limits": ["describes this trained network, not the airline system",
                   "an empty or shuffled lookback is outside what the network saw in training, so drops are an upper bound",
                   "one saved seed; seed spread is in the training decision file"],
        "final_test_used": False, "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(), "packages": package_versions(),
    }
    (tables / f"phase7a_sequence_perturbation_{arch}{infix}_decision.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote tables in {tables}. Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
