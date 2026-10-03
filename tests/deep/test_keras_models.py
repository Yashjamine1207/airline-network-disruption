"""Tests for the Keras models (mlp control, LSTM, GRU, compact Transformer).

These run real TensorFlow on tiny synthetic data, so they are skipped when TensorFlow is not
installed. The most important tests are the mask tests: a padded step must not change the
prediction, and a real step must.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from sklearn.metrics import roc_auc_score  # noqa: E402

from airline_disruption.deep import keras_models as km  # noqa: E402
from airline_disruption.deep.batching import CATEGORICAL_INPUT_NAMES, BatchSource  # noqa: E402
from airline_disruption.deep.config import DEFAULT_CONFIG  # noqa: E402
from airline_disruption.deep.inputs import ROLE_CODES, DeepInputs, SequenceScaler, StaticSpec  # noqa: E402
from airline_disruption.sequences.airport_bins import BinGrid  # noqa: E402
from airline_disruption.sequences.lookback import gather_sequences  # noqa: E402

LOOKBACK = 8
CHANNELS = 3
SMALL = {"units": 8, "dense": 16, "dropout": 0.1, "learning_rate": 3e-3}
VOCAB = {"origin_airport": 6, "destination_airport": 6, "route": 12, "carrier_identifier": 4}
GRID = BinGrid(start_seconds=0, bin_hours=3, n_bins=400)


def build(arch: str, seed: int = 1, config: dict | None = None, base_rate: float = 0.05):
    return km.build_model(arch, VOCAB, 10, LOOKBACK, CHANNELS, config or SMALL, base_rate, seed)


def random_batch(n: int = 32, seed: int = 0, n_real: int = 6) -> dict[str, np.ndarray]:
    """A model-ready batch with ``n_real`` real steps at the newest end and padding before them."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((n, LOOKBACK), dtype=bool)
    mask[:, LOOKBACK - n_real :] = True
    x = {
        "cat_origin": rng.integers(0, VOCAB["origin_airport"], n).astype("int32"),
        "cat_destination": rng.integers(0, VOCAB["destination_airport"], n).astype("int32"),
        "cat_route": rng.integers(0, VOCAB["route"], n).astype("int32"),
        "cat_carrier": rng.integers(0, VOCAB["carrier_identifier"], n).astype("int32"),
        "static_numeric": rng.normal(size=(n, 10)).astype("float32"),
        "sequence": np.where(mask[:, :, None], rng.normal(size=(n, LOOKBACK, CHANNELS)), 0.0).astype("float32"),
        "sequence_mask": mask,
    }
    return x


def predict(model, x: dict) -> np.ndarray:
    wanted = {k: v for k, v in x.items() if k in [i.name for i in model.inputs]}
    return model.predict(wanted, verbose=0).reshape(-1)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
SEQUENCE_ARCHS = ["lstm", "gru", "transformer"]


@pytest.mark.parametrize("arch", ["mlp", "lstm", "gru", "transformer"])
def test_each_architecture_builds_with_the_expected_inputs_and_a_probability_output(arch: str) -> None:
    model = build(arch)
    names = {i.name for i in model.inputs}
    expected = set(CATEGORICAL_INPUT_NAMES) | {"static_numeric"}
    if arch != "mlp":
        expected |= {"sequence", "sequence_mask"}
    assert names == expected
    p = predict(model, random_batch())
    assert p.shape == (32,) and ((p > 0) & (p < 1)).all()


def test_the_control_has_no_recurrent_layer_and_the_others_have_exactly_one() -> None:
    for arch, layer in (("mlp", None), ("lstm", "LSTM"), ("gru", "GRU"), ("transformer", None)):
        kinds = [type(x).__name__ for x in build(arch).layers]
        recurrent = [k for k in kinds if k in ("LSTM", "GRU")]
        assert recurrent == ([] if layer is None else [layer])


def test_the_transformer_is_one_compact_block_with_two_heads_and_no_recurrent_layer() -> None:
    model = build("transformer")
    kinds = [type(x).__name__ for x in model.layers]
    assert kinds.count("MultiHeadAttention") == 1
    assert kinds.count("LayerNormalization") == 2
    attention = model.get_layer("tf_attention_0")
    assert attention.num_heads == 2 and attention.key_dim == SMALL["units"] // 2
    assert model.get_layer("tf_position").input_dim == LOOKBACK


def test_the_transformer_needs_units_divisible_by_the_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build("transformer", config={**SMALL, "units": 7})


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        build("bert")
    with pytest.raises(ValueError, match="base_rate"):
        build("lstm", base_rate=0.0)
    with pytest.raises(ValueError, match="base_rate"):
        build("lstm", base_rate=1.0)


def test_parameter_count_orders_gru_below_lstm_and_control_below_both() -> None:
    n = {a: km.count_parameters(build(a)) for a in ("mlp", "lstm", "gru", "transformer")}
    assert n["mlp"] < n["gru"] < n["lstm"]
    assert n["transformer"] > n["mlp"]
    assert all(isinstance(v, int) for v in n.values())


# ---------------------------------------------------------------------------
# Initialisation and seeds
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch", ["mlp", "lstm", "gru", "transformer"])
def test_the_output_bias_starts_at_the_base_rate_log_odds(arch: str) -> None:
    rate = 0.03
    model = build(arch, base_rate=rate)
    bias = float(model.get_layer("probability").get_weights()[1][0])
    assert bias == pytest.approx(np.log(rate / (1 - rate)), abs=1e-6)
    # Untrained scores start near the base rate, not near 0.5.
    assert 0.3 * rate < predict(model, random_batch(256)).mean() < 3.0 * rate


@pytest.mark.parametrize("arch", ["mlp", "lstm", "gru", "transformer"])
def test_the_same_seed_gives_identical_initial_weights_and_a_different_seed_does_not(arch: str) -> None:
    a, b, c = build(arch, seed=5), build(arch, seed=5), build(arch, seed=6)
    same = all(np.array_equal(x, y) for x, y in zip(a.get_weights(), b.get_weights()))
    other = any(not np.array_equal(x, y) for x, y in zip(a.get_weights(), c.get_weights()))
    assert same and other


# ---------------------------------------------------------------------------
# Masking (the scope requires a test of masking behaviour)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_changing_a_padded_step_does_not_change_the_prediction(arch: str) -> None:
    model = build(arch)
    x = random_batch(n_real=5)
    before = predict(model, x)
    changed = {k: v.copy() for k, v in x.items()}
    changed["sequence"][:, : LOOKBACK - 5] = 123.0  # the padded (oldest) steps
    assert np.allclose(before, predict(model, changed), atol=1e-6)


@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_changing_a_real_step_does_change_the_prediction(arch: str) -> None:
    model = build(arch)
    x = random_batch(n_real=5)
    before = predict(model, x)
    changed = {k: v.copy() for k, v in x.items()}
    changed["sequence"][:, -1] += 3.0  # the newest real step
    assert np.abs(before - predict(model, changed)).max() > 1e-4


@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_masked_steps_are_skipped_not_just_zero(arch: str) -> None:
    """The mask must be used. Without it the model would read the padded steps as data.

    Padded steps are given a non-zero value here on purpose. A fresh LSTM or GRU has zero biases,
    so leading zeros leave its state at zero and would hide a missing mask.
    """
    model = build(arch)
    with_mask = random_batch(n_real=4)
    with_mask["sequence"][:, : LOOKBACK - 4] = 2.5
    unmasked = {k: v.copy() for k, v in with_mask.items()}
    unmasked["sequence_mask"][:] = True  # same values, but padding now counts as data
    assert np.abs(predict(model, with_mask) - predict(model, unmasked)).max() > 1e-4


@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_a_fully_masked_sequence_gives_a_finite_probability(arch: str) -> None:
    """Rows in the first 24 hours of the file have no real bin at all."""
    model = build(arch)
    x = random_batch(n_real=0)
    p = predict(model, x)
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    # A hidden NaN would NOT show up as NaN here: Keras turns relu(NaN) into 0, which makes every row
    # score exactly the base rate. The static inputs are random, so a healthy model must vary between rows.
    assert p.std() > 1e-5, "all rows scored the same: the static inputs were lost (a hidden NaN?)"
    # ...and the sequence values are irrelevant, so the score depends on the static inputs only.
    x2 = {k: v.copy() for k, v in x.items()}
    x2["sequence"][:] = 55.0
    assert np.allclose(p, predict(model, x2), atol=1e-6)


def test_transformer_padded_positions_cannot_leak_through_the_position_embedding() -> None:
    """Padded steps also carry a position embedding. Changing those rows must not move the score."""
    model = build("transformer")
    x = random_batch(n_real=5)
    before = predict(model, x)
    table = model.get_layer("tf_position")
    weights = table.get_weights()[0].copy()
    weights[: LOOKBACK - 5] = 9.0  # the embeddings of the padded (oldest) positions
    table.set_weights([weights])
    assert np.allclose(before, predict(model, x), atol=1e-5)


def test_transformer_real_steps_never_read_padded_steps_but_do_read_each_other() -> None:
    """Attention must ignore padded keys and use real ones: only the newest real step differs between these rows."""
    model = build("transformer")
    x = random_batch(n_real=5)
    baseline = predict(model, x)
    other = {k: v.copy() for k, v in x.items()}
    other["sequence"][:, LOOKBACK - 5] += 2.0  # the OLDEST real step (all later steps could attend to it)
    assert np.abs(baseline - predict(model, other)).max() > 1e-4


def test_the_control_ignores_any_sequence() -> None:
    """The mlp has no sequence input at all, so a sequence cannot leak into it."""
    model = build("mlp")
    assert "sequence" not in {i.name for i in model.inputs}


@pytest.mark.parametrize("arch", SEQUENCE_ARCHS)
def test_the_sequence_layer_is_sensitive_to_the_order_of_real_steps(arch: str) -> None:
    model = build(arch)
    x = random_batch(n_real=6, seed=3)
    reversed_x = {k: v.copy() for k, v in x.items()}
    reversed_x["sequence"][:, LOOKBACK - 6 :] = x["sequence"][:, LOOKBACK - 6 :][:, ::-1]
    assert np.abs(predict(model, x) - predict(model, reversed_x)).max() > 1e-4


# ---------------------------------------------------------------------------
# Data-fed tests: a synthetic problem whose answer is in the sequence
# ---------------------------------------------------------------------------
def synthetic_inputs(n: int = 2400, seed: int = 0) -> tuple[DeepInputs, np.ndarray]:
    """Rows whose label is 'the newest closed bin had many departures'. Static inputs are pure noise."""
    rng = np.random.default_rng(seed)
    table = rng.poisson(6.0, size=(4, GRID.n_bins, CHANNELS)).astype("float32")
    airport = rng.integers(0, 4, n).astype("int16")
    end_bin = rng.integers(30, GRID.n_bins, n).astype("int32")
    values, mask = gather_sequences(table, airport, end_bin, GRID, LOOKBACK)
    label = (values[:, -1, 0] >= 8).astype("int8")  # newest bin, departures channel
    assert mask[:, -1].all() and 0.2 < label.mean() < 0.6
    role = np.where(np.arange(n) < int(0.7 * n), ROLE_CODES["fit"], ROLE_CODES["validation"]).astype("int8")
    inputs = DeepInputs(
        codes=rng.integers(1, 5, size=(n, 4)).astype("int32"),
        numeric=rng.normal(size=(n, 10)).astype("float32"),
        label=label,
        role=role,
        airport_index=airport,
        end_bin=end_bin,
        source_row_number=np.arange(n, dtype="int64") + 5000,
        prediction_seconds=np.arange(n, dtype="int64") * 60,
        spec=StaticSpec({}, {}, {}, {}),
        scaler=SequenceScaler([1.8, 1.8, 1.8], [0.5, 0.5, 0.5]),
    )
    return inputs, table


def sources(arch: str, batch_size: int = 128, shuffle_fit: bool = True):
    inputs, table = synthetic_inputs()
    use_seq = arch != "mlp"
    common = dict(table=table if use_seq else None, grid=GRID if use_seq else None, lookback=LOOKBACK, use_sequence=use_seq)
    fit = BatchSource(inputs, inputs.rows("fit"), batch_size=batch_size, shuffle=shuffle_fit, seed=3, **common)
    valid = BatchSource(inputs, inputs.rows("validation"), batch_size=batch_size, **common)
    return inputs, fit, valid


def fit_small(arch: str, seed: int = 1, epochs: int = 12):
    inputs, fit, valid = sources(arch)
    model = km.build_model(arch, {c: 5 for c in VOCAB}, 10, LOOKBACK, CHANNELS, SMALL, float(inputs.label[fit.rows].mean()), seed)
    info = km.train_model(model, fit, valid, max_epochs=epochs, patience=4)
    return model, info, valid


@pytest.mark.parametrize("arch", ["lstm", "gru"])
def test_recurrent_models_learn_a_signal_that_only_the_sequence_holds(arch: str) -> None:
    model, info, valid = fit_small(arch)
    auc = roc_auc_score(valid.labels(), km.predict_scores(model, valid))
    assert auc > 0.85, f"{arch} did not find a clear signal in the sequence (AUC {auc:.3f})"
    assert info["best_epoch"] <= info["epochs_run"] <= 12


def test_the_transformer_learns_a_signal_that_only_the_sequence_holds() -> None:
    """The bar is lower than for the recurrent models: one block, mean pooling and only 1,680 rows."""
    model, info, valid = fit_small("transformer", epochs=25)
    auc = roc_auc_score(valid.labels(), km.predict_scores(model, valid))
    assert auc > 0.7, f"the transformer did not find the signal in the sequence (AUC {auc:.3f})"


def test_the_tabular_control_cannot_find_a_signal_that_only_the_sequence_holds() -> None:
    """If this failed, the control would be leaking the sequence, which would spoil the ablation."""
    model, _, valid = fit_small("mlp")
    auc = roc_auc_score(valid.labels(), km.predict_scores(model, valid))
    assert auc < 0.6, f"the mlp control scored AUC {auc:.3f} on noise"


def test_training_restores_the_best_epoch_weights() -> None:
    """Early-stopping rows follow the OPPOSITE rule to the fit rows, so every epoch of learning makes them worse.

    The best epoch is then the first one and the last epoch is clearly worse. A run that failed
    to restore the best weights would end with the last epoch's (worse) score.
    """
    inputs, table = synthetic_inputs()
    flipped_label = inputs.label.copy()
    stop = inputs.rows("validation")
    flipped_label[stop] = 1 - flipped_label[stop]
    flipped = DeepInputs(**{**inputs.__dict__, "label": flipped_label})
    common = dict(table=table, grid=GRID, lookback=LOOKBACK, batch_size=128)
    fit_source = BatchSource(flipped, flipped.rows("fit"), shuffle=True, seed=3, **common)
    stop_source = BatchSource(flipped, stop, **common)
    model = km.build_model("gru", {c: 5 for c in VOCAB}, 10, LOOKBACK, CHANNELS, SMALL, float(flipped.label[fit_source.rows].mean()), 1)
    info = km.train_model(model, fit_source, stop_source, max_epochs=10, patience=2)

    curve = info["history"]["val_pr_auc"]
    assert len(curve) == info["epochs_run"] and info["best_epoch"] < info["epochs_run"]
    assert curve[-1] < info["best_stop_pr_auc"] - 0.02, "the scenario should make the last epoch clearly worse than the best"
    now = model.evaluate(km.make_dataset(stop_source, follow_shuffle=False), verbose=0, return_dict=True)["pr_auc"]
    assert now == pytest.approx(info["best_stop_pr_auc"], abs=2e-3)  # the best epoch's weights are back in place
    assert now > curve[-1] + 0.02


def test_predict_scores_line_up_with_the_source_rows_even_when_the_source_shuffles() -> None:
    inputs, fit, _ = sources("lstm", shuffle_fit=True)
    model = km.build_model("lstm", {c: 5 for c in VOCAB}, 10, LOOKBACK, CHANNELS, SMALL, 0.3, 4)
    scores = km.predict_scores(model, fit)
    assert len(scores) == len(fit.rows)
    # Score each row on its own and compare: a shuffled prediction order would not match.
    for position in (0, 7, len(fit.rows) - 1):
        one = BatchSource(inputs, fit.rows[[position]], fit.table, fit.grid, batch_size=1, lookback=LOOKBACK)
        assert km.predict_scores(model, one)[0] == pytest.approx(scores[position], abs=1e-5)


def test_predictions_do_not_depend_on_which_other_rows_share_a_batch() -> None:
    inputs, fit, _ = sources("gru", shuffle_fit=False)
    model = km.build_model("gru", {c: 5 for c in VOCAB}, 10, LOOKBACK, CHANNELS, SMALL, 0.3, 4)
    rows = fit.rows[:200]
    a = km.predict_scores(model, BatchSource(inputs, rows, fit.table, fit.grid, batch_size=200, lookback=LOOKBACK))
    b = km.predict_scores(model, BatchSource(inputs, rows, fit.table, fit.grid, batch_size=17, lookback=LOOKBACK))
    assert np.allclose(a, b, atol=1e-5)


def test_training_twice_with_the_same_seed_gives_nearly_the_same_model() -> None:
    """Records how repeatable a fit is on CPU. Small float differences between runs are allowed."""
    m1, _, valid = fit_small("lstm", seed=9, epochs=4)
    m2, _, _ = fit_small("lstm", seed=9, epochs=4)
    assert np.allclose(km.predict_scores(m1, valid), km.predict_scores(m2, valid), atol=5e-3)


def test_a_non_finite_prediction_is_rejected() -> None:
    inputs, fit, _ = sources("mlp", shuffle_fit=False)
    model = km.build_model("mlp", {c: 5 for c in VOCAB}, 10, LOOKBACK, CHANNELS, SMALL, 0.3, 4)
    bad = BatchSource(inputs, fit.rows[:10], None, None, batch_size=10, use_sequence=False)
    bad.inputs = DeepInputs(**{**inputs.__dict__, "numeric": np.full_like(inputs.numeric, np.nan)})
    with pytest.raises(ValueError, match="non-finite"):
        km.predict_scores(model, bad)


def test_the_default_config_builds_every_architecture() -> None:
    for arch in ("mlp", "lstm", "gru", "transformer"):
        assert km.count_parameters(km.build_model(arch, VOCAB, 10, 16, 3, DEFAULT_CONFIG, 0.05, 1)) > 0


@pytest.mark.parametrize("arch", ["mlp", "gru", "transformer"])
def test_a_saved_model_loads_back_and_predicts_the_same_scores(arch: str, tmp_path) -> None:
    model = build(arch, seed=3)
    x = random_batch(40, seed=8)
    path = tmp_path / f"{arch}.keras"
    km.save_model(model, path)
    reloaded = km.load_model(path)
    assert np.allclose(predict(model, x), predict(reloaded, x), atol=1e-6)
    km.release()  # must not raise, and the reloaded model keeps working afterwards
    assert np.isfinite(predict(reloaded, x)).all()
