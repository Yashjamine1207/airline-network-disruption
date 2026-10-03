#!/usr/bin/env python
"""Phase 6, Steps 7b and 9: train the MLP control, the LSTM, the GRU or the compact Transformer on the airport schedule sequence.

Save as:   scripts/train_phase6_sequence_model.py
Run from the project root, after build_phase6_deep_inputs.py and the Step 6 LightGBM run:

    # 1. Smoke run (a few minutes). Tagged "smoke"; numbers are not results.
    python scripts/train_phase6_sequence_model.py --arch lstm --max-fit-rows 200000 --max-epochs 2 --extra-seeds 0 --bootstrap 20

    # 2. Real runs. Do the control first: it is the yardstick for the sequence models.
    python scripts/train_phase6_sequence_model.py --arch mlp
    python scripts/train_phase6_sequence_model.py --arch lstm --control-predictions data/processed/phase6/predictions/mlp_validation.parquet
    python scripts/train_phase6_sequence_model.py --arch gru  --control-predictions data/processed/phase6/predictions/mlp_validation.parquet

    # More configurations (DEC-020 allows up to 5 random draws besides the default).
    # Use a tag so the default-only files above are not overwritten:
    python scripts/train_phase6_sequence_model.py --arch lstm --n-configs 5 --tag tuned5

    # Step 9: the compact Transformer (DEC-022), compared with the MLP control, the LSTM and tuned LightGBM.
    python scripts/train_phase6_sequence_model.py --arch transformer --control-predictions data/processed/phase6/predictions/mlp_validation.parquet --recurrent-predictions data/processed/phase6/predictions/lstm_validation.parquet

    # The 2020 ablation (DEC-021): train without 2020, and on the same number of random rows.
    # Compare each with the full-fit run of the same architecture (passed as the control).
    # One line per command, so it pastes into PowerShell:
    python scripts/train_phase6_sequence_model.py --arch lstm --fit-variant no2020 --control-predictions data/processed/phase6/predictions/lstm_validation.parquet --control-column lstm --control-label "lstm, full fit"
    python scripts/train_phase6_sequence_model.py --arch lstm --fit-variant matched --control-predictions data/processed/phase6/predictions/lstm_validation.parquet --control-column lstm --control-label "lstm, full fit"

Protocol (DEC-019 and DEC-020, fixed before any result)
-------------------------------------------------------
* FIT rows train the network. EARLY-STOP rows (development, from 2021-07-01 UTC) choose the epoch.
  The 2022 validation rows play no part in training or stopping.
* The default configuration is always trained first. Random draws (``--n-configs``) come from a
  small pre-registered space. The challenger is the draw with the best PR-AUC on the FIRST half of
  2022; it is adopted only if a paired day-level bootstrap on the SECOND half puts the whole 95%
  interval above the default. Otherwise the default stays.
* The chosen configuration is trained again with ``--extra-seeds`` more seeds, to show how much a
  score moves with the seed alone.
* The result is compared with the tuned LightGBM benchmark (and with the MLP control, if given) on
  the second half of 2022, where no selection took place, and on the full year for context.
* 2023 is never loaded. The inputs file contains no final-test rows.

The MLP control has the same static inputs and head but no sequence. The gap between a sequence
model and the control is the value of the sequence; the gap between the control and LightGBM is
the cost or benefit of using a neural network on these features at all.

Writes (tables are small and safe to commit; models and predictions stay out of Git):
    reports/tables/phase6_deep_<arch>[_tag]_{configs,history,validation,ranking,calibration,comparisons}.csv
    reports/tables/phase6_deep_<arch>[_tag]_decision.json
    configs/deep_<arch>_selected.json                     (plain full run only)
    data/processed/phase6/predictions/<arch>_validation[_tag].parquet
    data/processed/phase6/models/<arch>[_tag]_c<config>_s<seed>.keras
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
    from airline_disruption.deep import config as cfg
    from airline_disruption.deep import inputs as di
    from airline_disruption.deep import protocol as proto
    from airline_disruption.deep.batching import BatchSource
    from airline_disruption.evaluation.classification import (
        DEFAULT_FRACTIONS,
        binary_metrics,
        calibration_table,
        paired_day_bootstrap,
        ranking_table,
    )
    from airline_disruption.features.feature_sets import FEATURE_VERSION
    from airline_disruption.models import lightgbm_tuning as tune
    from airline_disruption.sequences import airport_bins as ab
    from airline_disruption.utils.run_info import hardware_context, package_versions
except ImportError:  # package not installed in editable mode: fall back to the src folder
    sys.path.insert(0, str(ROOT / "src"))
    from airline_disruption.deep import config as cfg  # noqa: E402
    from airline_disruption.deep import inputs as di  # noqa: E402
    from airline_disruption.deep import protocol as proto  # noqa: E402
    from airline_disruption.deep.batching import BatchSource  # noqa: E402
    from airline_disruption.evaluation.classification import (  # noqa: E402
        DEFAULT_FRACTIONS,
        binary_metrics,
        calibration_table,
        paired_day_bootstrap,
        ranking_table,
    )
    from airline_disruption.features.feature_sets import FEATURE_VERSION  # noqa: E402
    from airline_disruption.models import lightgbm_tuning as tune  # noqa: E402
    from airline_disruption.sequences import airport_bins as ab  # noqa: E402
    from airline_disruption.utils.run_info import hardware_context, package_versions  # noqa: E402

INPUTS_DIR = "data/processed/phase6/deep_inputs_v1"
TABLE_PATH = "data/processed/phase6/airport_bins_v1.npz"
BENCHMARK_PATH = "data/processed/phase6/predictions/lightgbm_validation_phase6_tuned_ladder.parquet"
BENCHMARK_COLUMN = "lightgbm_base_no_year"
MAX_DRAWS = 5  # DEC-020


def fail(message: str) -> None:
    print(f"\nCHECK FAILED: {message}")
    raise SystemExit(1)


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_metrics(y, scores)["pr_auc"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 sequence model training (validation only)")
    parser.add_argument("--arch", required=True, choices=cfg.ARCHITECTURES)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inputs", default=INPUTS_DIR, help="folder written by build_phase6_deep_inputs.py")
    parser.add_argument("--n-configs", type=int, default=0, help=f"random draws besides the default (at most {MAX_DRAWS})")
    parser.add_argument("--extra-seeds", type=int, default=2, help="extra seeds for the chosen configuration (0 to skip)")
    parser.add_argument("--n-threads", type=int, default=8, help="TensorFlow CPU threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=ab.LOOKBACK_BINS, help="bins used by the recurrent layer (at most 16)")
    parser.add_argument("--max-fit-rows", type=int, default=None, help="subsample fit rows, smoke runs only")
    parser.add_argument("--max-epochs", type=int, default=None, help="override the epoch limit, smoke runs only")
    parser.add_argument("--bootstrap", type=int, default=200, help="day-level bootstrap resamples (0 to skip)")
    parser.add_argument("--fit-variant", default="full", choices=proto.FIT_VARIANTS,
                        help="rows the network trains on: full, no2020 (2020 removed) or matched (random rows, same count as no2020)")
    parser.add_argument("--tag", default="")
    parser.add_argument("--benchmark-predictions", default=BENCHMARK_PATH)
    parser.add_argument("--benchmark-column", default=BENCHMARK_COLUMN)
    parser.add_argument("--control-predictions", default=None, help="predictions file of the MLP control run (optional)")
    parser.add_argument("--control-column", default="mlp")
    parser.add_argument("--recurrent-predictions", default=None,
                        help="predictions file of the best recurrent model, for the compact Transformer's extra comparison (DEC-022)")
    parser.add_argument("--recurrent-column", default="lstm")
    parser.add_argument("--recurrent-label", default=None, help="name shown for that model (default: the column name)")
    parser.add_argument("--control-label", default=None,
                        help="name shown for the control in the printout (default: the MLP control; for the 2020 ablation pass e.g. 'lstm, full fit')")
    args = parser.parse_args()
    root = args.root.resolve()
    started = time.perf_counter()
    arch = args.arch
    sequence_model = arch != "mlp"

    if not 0 <= args.n_configs <= MAX_DRAWS:
        fail(f"--n-configs must be between 0 and {MAX_DRAWS} (DEC-020)")
    if not 1 <= args.lookback <= ab.LOOKBACK_BINS:
        fail(f"--lookback must be between 1 and {ab.LOOKBACK_BINS}")
    for path in (root / args.inputs, root / TABLE_PATH, root / args.benchmark_predictions):
        if not path.exists():
            fail(f"not found: {path}  (run build_phase6_deep_inputs.py and the Step 6 LightGBM run first)")

    smoke = args.max_fit_rows is not None or args.max_epochs is not None
    official = not smoke and not args.tag and args.fit_variant == "full"
    tag_parts = [args.tag] if args.tag else (["smoke"] if smoke else [])
    if args.fit_variant != "full":
        tag_parts.append(args.fit_variant)  # an ablation run never overwrites the full-fit files
    tag = "_".join(tag_parts)
    suffix = arch + (f"_{tag}" if tag else "")
    tables = root / "reports/tables"
    models_dir = root / "data/processed/phase6/models"
    predictions_dir = root / "data/processed/phase6/predictions"
    for folder in (tables, models_dir, predictions_dir):
        folder.mkdir(parents=True, exist_ok=True)

    try:
        from airline_disruption.deep import keras_models as km
    except ImportError as problem:
        fail(f"TensorFlow is needed for this script ({problem})")
    km.set_threads(args.n_threads)

    print(f"Architecture: {arch}   configurations: default + {args.n_configs} draws   extra seeds: {args.extra_seeds}   tag: {tag or '-'}")
    if smoke:
        print("SMOKE RUN: reduced rows or epochs. Numbers are not results and no selection file is written.")

    # ------------------------------------------------------------------
    # Inputs and rows
    # ------------------------------------------------------------------
    inputs = di.load_deep_inputs(root / args.inputs)
    manifest = inputs.manifest
    if manifest.get("input_version") != di.INPUT_VERSION or manifest.get("sequence_version") != ab.SEQUENCE_VERSION:
        fail("the inputs folder was built by a different code version; rerun build_phase6_deep_inputs.py")
    if manifest.get("final_test_used") is not False:
        fail("the inputs manifest does not say the final test was excluded")
    table, airports, grid = ab.load_bin_table(root / TABLE_PATH)

    fit_all, stop_rows, val_rows = inputs.rows("fit"), inputs.rows("early_stop"), inputs.rows("validation")
    seconds = inputs.prediction_seconds
    fit_rows = proto.subsample_rows(fit_all, args.max_fit_rows, args.seed)
    try:
        fit_rows = proto.fit_variant_rows(fit_rows, seconds, args.fit_variant, args.seed)
    except ValueError as problem:
        fail(str(problem))
    if not (seconds[fit_rows].max() < seconds[stop_rows].min() and seconds[stop_rows].max() < seconds[val_rows].min()):
        fail("fit, early-stop and validation rows are not in chronological order")
    try:
        proto.require_two_classes(inputs.label[fit_rows], "fit")
        proto.require_two_classes(inputs.label[stop_rows], "early-stopping")
        proto.require_two_classes(inputs.label[val_rows], "validation")
    except ValueError as problem:
        fail(str(problem))
    print(f"  fit {len(fit_rows):,} rows (variant: {args.fit_variant}) | early stopping {len(stop_rows):,} | validation {len(val_rows):,}")
    if args.fit_variant != "full":
        years = pd.Series(proto.calendar_year_utc(seconds[fit_rows])).value_counts().sort_index()
        print("  fit rows by UTC year: " + ", ".join(f"{y} {n:,}" for y, n in years.items()))

    y_val = inputs.label[val_rows]
    stamps = pd.Series(pd.to_datetime(seconds[val_rows], unit="s", utc=True))
    try:
        selection, confirmation = tune.validation_halves(stamps)
    except ValueError as problem:
        fail(str(problem))
    days = stamps.dt.floor("D").to_numpy()
    everything = np.ones(len(y_val), dtype=bool)
    print(f"  validation halves: selection {int(selection.sum()):,} rows, confirmation {int(confirmation.sum()):,} rows")

    # LightGBM benchmark (and the control) scored the same validation rows.
    ids_val = inputs.source_row_number[val_rows]
    try:
        benchmark = proto.align_scores(ids_val, y_val, pd.read_parquet(root / args.benchmark_predictions), args.benchmark_column)
        control = None
        if args.control_predictions:
            control = proto.align_scores(ids_val, y_val, pd.read_parquet(root / args.control_predictions), args.control_column)
        recurrent = None
        if args.recurrent_predictions:
            recurrent = proto.align_scores(ids_val, y_val, pd.read_parquet(root / args.recurrent_predictions), args.recurrent_column)
    except ValueError as problem:
        fail(str(problem))
    print(f"  benchmark ({args.benchmark_column}) PR-AUC: selection half {pr_auc(y_val[selection], benchmark[selection]):.5f}, "
          f"confirmation half {pr_auc(y_val[confirmation], benchmark[confirmation]):.5f}")

    max_epochs = args.max_epochs or cfg.TRAINING["max_epochs"]
    batch_size, patience = cfg.TRAINING["batch_size"], cfg.TRAINING["patience"]
    base_rate = float(inputs.label[fit_rows].mean())
    sizes = inputs.spec.vocabulary_sizes()

    def fit_and_score(config: dict, seed: int, name: str):
        """Train one network, save it, return ``(info, validation scores, model file, parameter count)``."""
        common = dict(
            table=table if sequence_model else None,
            grid=grid if sequence_model else None,
            batch_size=batch_size,
            lookback=args.lookback,
            use_sequence=sequence_model,
        )
        fit_source = BatchSource(inputs, fit_rows, shuffle=True, seed=seed, **common)
        stop_source = BatchSource(inputs, stop_rows, **common)
        val_source = BatchSource(inputs, val_rows, **common)
        model = km.build_model(arch, sizes, len(di.NUMERIC_NAMES), args.lookback, len(ab.CHANNELS), config, base_rate, seed)
        n_params = km.count_parameters(model)
        info = km.train_model(model, fit_source, stop_source, max_epochs, patience)
        scores = km.predict_scores(model, val_source)
        path = models_dir / f"{suffix}_{name}_s{seed}.keras"
        km.save_model(model, path)
        km.release()
        return info, scores, path, n_params

    # ------------------------------------------------------------------
    # Default configuration, then random draws
    # ------------------------------------------------------------------
    configs = cfg.sample_configs(args.n_configs, args.seed)
    rows, history_rows = [], []
    scores_by_config: dict[int, np.ndarray] = {}
    files_by_config: dict[int, Path] = {}
    infos: dict[int, dict] = {}
    params_by_config: dict[int, int] = {}
    for number, config in enumerate(configs):
        label = "default" if number == 0 else f"draw {number}"
        print(f"\n=== [{number}/{args.n_configs}] {label}: {config} ===", flush=True)
        info, scores, path, n_params = fit_and_score(config, args.seed, f"c{number}")
        scores_by_config[number], files_by_config[number], infos[number], params_by_config[number] = scores, path, info, n_params
        row = {
            "config": number,
            "label": label,
            "overrides": json.dumps(config),
            "parameters": n_params,
            "best_epoch": info["best_epoch"],
            "epochs_run": info["epochs_run"],
            "train_seconds": round(info["seconds"], 1),
            "stop_pr_auc_keras": round(info["best_stop_pr_auc"], 6),
            "pr_auc_selection_half": pr_auc(y_val[selection], scores[selection]),
            "pr_auc_confirmation_half": pr_auc(y_val[confirmation], scores[confirmation]),
            "pr_auc_full_year": pr_auc(y_val, scores),
        }
        rows.append(row)
        print(f"  -> best epoch {row['best_epoch']} of {row['epochs_run']}, {n_params:,} parameters, {info['seconds']:,.0f} s; "
              f"validation PR-AUC selection {row['pr_auc_selection_half']:.5f}  confirmation {row['pr_auc_confirmation_half']:.5f}", flush=True)
        for epoch in range(info["epochs_run"]):
            history_rows.append({"config": number, "seed": args.seed, "epoch": epoch + 1,
                                 **{k: values[epoch] for k, values in info["history"].items()}})
        pd.DataFrame(rows).to_csv(tables / f"phase6_deep_{suffix}_configs.csv", index=False)  # keep progress if interrupted
    results = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Select on the first half, confirm on the second
    # ------------------------------------------------------------------
    best, adopt, tuning_outcome = 0, False, None
    if args.n_configs > 0:
        best = tune.best_config_index(results["pr_auc_selection_half"].to_numpy())
        print(f"\nChallenger (best selection-half PR-AUC): config {best}  {configs[best]}")
        if args.bootstrap > 0:
            tuning_outcome = paired_day_bootstrap(
                y_val[confirmation], scores_by_config[0][confirmation], scores_by_config[best][confirmation],
                days[confirmation], n_boot=args.bootstrap, seed=args.seed)
            adopt = tune.decide_adoption(tuning_outcome["ci_low"])
            print(f"  confirmation half: default {tuning_outcome['pr_auc_reference']:.5f}   challenger {tuning_outcome['pr_auc_challenger']:.5f}   "
                  f"difference {tuning_outcome['difference']:+.5f}  95% interval [{tuning_outcome['ci_low']:+.5f}, {tuning_outcome['ci_high']:+.5f}]")
        else:
            print("  bootstrap skipped: the default is kept.")
        print(f"  DECISION: {'adopt the challenger' if adopt else 'keep the default configuration'}")
    final = best if adopt else 0
    final_config, final_scores = configs[final], scores_by_config[final]
    print(f"\nFinal configuration: config {final} {final_config}")

    # ------------------------------------------------------------------
    # Seed noise
    # ------------------------------------------------------------------
    seeds = [args.seed]
    seed_scores = {args.seed: final_scores}
    for extra in range(1, args.extra_seeds + 1):
        seed = args.seed + extra
        print(f"\n=== seed check: config {final}, seed {seed} ===", flush=True)
        info, scores, _, _ = fit_and_score(final_config, seed, f"c{final}")
        seeds.append(seed)
        seed_scores[seed] = scores
        for epoch in range(info["epochs_run"]):
            history_rows.append({"config": final, "seed": seed, "epoch": epoch + 1, **{k: v[epoch] for k, v in info["history"].items()}})
    seed_h2 = [pr_auc(y_val[confirmation], seed_scores[s][confirmation]) for s in seeds]
    if len(seeds) > 1:
        print("\nSeed check (confirmation-half PR-AUC): " + "  ".join(f"seed {s}: {v:.5f}" for s, v in zip(seeds, seed_h2))
              + f"   spread {max(seed_h2) - min(seed_h2):.5f}")
    pd.DataFrame(history_rows).to_csv(tables / f"phase6_deep_{suffix}_history.csv", index=False)

    # ------------------------------------------------------------------
    # Results for the chosen model
    # ------------------------------------------------------------------
    periods = {"2022 second half (confirmation)": confirmation, "2022 full year": everything, "2022 first half (selection)": selection}
    metric_rows = []
    for period, mask in periods.items():
        m = binary_metrics(y_val[mask], final_scores[mask])
        top = ranking_table(y_val[mask], final_scores[mask]).set_index("fraction")
        metric_rows.append({"model": arch, "period": period, **m,
                            "precision_at_1pct": top.loc[0.01, "precision"], "recall_at_1pct": top.loc[0.01, "recall"]})
        bench = binary_metrics(y_val[mask], benchmark[mask])
        metric_rows.append({"model": args.benchmark_column, "period": period, **bench})
    metrics = pd.DataFrame(metric_rows)
    print("\nValidation results (2022):")
    print(metrics[["model", "period", "pr_auc", "pr_auc_lift", "roc_auc", "brier_score"]].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ------------------------------------------------------------------
    # Paired comparisons
    # ------------------------------------------------------------------
    comparisons = []
    references = [(args.benchmark_column, benchmark)]
    if control is not None:
        references.append((args.control_label or f"{args.control_column} (control, no sequence)", control))
    if recurrent is not None:
        references.append((args.recurrent_label or f"{args.recurrent_column} (best recurrent model)", recurrent))
    if args.bootstrap > 0:
        for reference_name, reference_scores in references:
            for period in ("2022 second half (confirmation)", "2022 full year"):
                mask = periods[period]
                outcome = paired_day_bootstrap(y_val[mask], reference_scores[mask], final_scores[mask], days[mask],
                                               n_boot=args.bootstrap, seed=args.seed)
                sentence = proto.verdict(outcome["difference"], outcome["ci_low"], outcome["ci_high"], arch, reference_name)
                comparisons.append({"challenger": arch, "reference": reference_name, "period": period, **outcome, "verdict": sentence})
                print(f"\n{arch} minus {reference_name}, {period}:\n  {outcome['pr_auc_challenger']:.5f} vs {outcome['pr_auc_reference']:.5f}  "
                      f"difference {outcome['difference']:+.5f}  95% interval [{outcome['ci_low']:+.5f}, {outcome['ci_high']:+.5f}]\n  {sentence}")
        print("\nThe second-half rows are the clean comparison: no setting of either model was chosen on them. "
              "The first half was used to choose LightGBM's settings (and the network's, if random draws were tried), "
              "so the first half and the full year favour both models a little.")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    primary = infos[final]
    model_file = files_by_config[final]
    predictions = pd.DataFrame({
        "source_row_number": ids_val,
        "prediction_day_utc": days,
        "label": y_val,
        arch: final_scores,
        **{f"{arch}_seed{s}": seed_scores[s] for s in seeds[1:]},
    })
    predictions.to_parquet(predictions_dir / f"{arch}_validation{'_' + tag if tag else ''}.parquet", index=False)

    metrics.to_csv(tables / f"phase6_deep_{suffix}_validation.csv", index=False)
    ranking_table(y_val, final_scores).to_csv(tables / f"phase6_deep_{suffix}_ranking.csv", index=False)
    calibration_table(y_val, final_scores).to_csv(tables / f"phase6_deep_{suffix}_calibration.csv", index=False)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(tables / f"phase6_deep_{suffix}_comparisons.csv", index=False)

    decision = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": arch,
        "tag": tag,
        "smoke_run": smoke,
        "target": "severe_delay_120",
        "prediction_time": "scheduled departure UTC minus 2 hours",
        "input_version": di.INPUT_VERSION,
        "feature_version": FEATURE_VERSION,
        "static_inputs": "base_no_year (DEC-018)",
        "sequence": {
            "version": ab.SEQUENCE_VERSION,
            "entity": "origin airport schedule (DEC-017)" if sequence_model else "none (control)",
            "lookback_bins": args.lookback if sequence_model else None,
            "lookback_hours": args.lookback * ab.BIN_HOURS if sequence_model else None,
            "bin_hours": ab.BIN_HOURS,
            "channels": list(ab.CHANNELS),
            "padding": "zeros, masked; the mask is passed to the sequence layer (recurrent layer, or attention keys and the final average)",
            "outcome_columns_read": False,
        },
        "protocol": "phase6 (fit < 2021-07-01 <= early stop; validation = 2022)",
        "fit_variant": args.fit_variant,
        "fit_rows_by_utc_year": {int(y): int(n) for y, n in pd.Series(proto.calendar_year_utc(seconds[fit_rows])).value_counts().sort_index().items()},
        "training": {"batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
                     "loss": "binary cross-entropy, unweighted", "optimiser": "Adam",
                     "early_stopping_monitor": "val_pr_auc on the early-stop rows"},
        "rows": {"fit": int(len(fit_rows)), "early_stop": int(len(stop_rows)), "validation": int(len(val_rows))},
        "n_random_draws": args.n_configs,
        "search_space": cfg.SEARCH_SPACE,
        "default_config": cfg.DEFAULT_CONFIG,
        "seed": args.seed,
        "n_threads": args.n_threads,
        "selection_period": "2022, prediction time before 2022-07-01 UTC",
        "confirmation_period": "2022, prediction time from 2022-07-01 UTC",
        "challenger_config": int(best) if args.n_configs > 0 else None,
        "challenger_overrides": configs[best] if args.n_configs > 0 else None,
        "challenger_adopted": bool(adopt),
        "confirmation_bootstrap": tuning_outcome,
        "final_config_index": int(final),
        "final_config": final_config,
        "parameter_count": params_by_config[final],
        "primary_train_seconds": round(primary["seconds"], 1),
        "primary_best_epoch": primary["best_epoch"],
        "primary_epochs_run": primary["epochs_run"],
        "seed_check": {"seeds": seeds, "pr_auc_confirmation_half": seed_h2,
                       "spread": float(max(seed_h2) - min(seed_h2))},
        "benchmark": {"file": args.benchmark_predictions, "column": args.benchmark_column},
        "extra_references": {"control": args.control_predictions, "recurrent": args.recurrent_predictions},
        "comparisons": comparisons,
        "model_file": str(model_file.relative_to(root)),
        "review_fractions": list(DEFAULT_FRACTIONS),
        "final_test_used": False,
        "total_seconds": round(time.perf_counter() - started, 1),
        "hardware": hardware_context(),
        "packages": package_versions(),
    }
    (tables / f"phase6_deep_{suffix}_decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    if official:
        (root / "configs").mkdir(exist_ok=True)
        (root / f"configs/deep_{arch}_selected.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote configs/deep_{arch}_selected.json")
    print(f"\nWrote tables in {tables} (files phase6_deep_{suffix}_*), predictions in {predictions_dir} and models in {models_dir}.")
    print(f"Final test was not loaded. Done in {time.perf_counter() - started:,.0f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
