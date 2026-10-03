"""TensorFlow/Keras models for Phase 6: tabular-only control, LSTM, GRU and a compact Transformer.

Phase 6, Steps 7 and 9. TensorFlow is imported inside the functions so the rest of the
package works without it.

Architectures (``config.ARCHITECTURES``)
-----------------------------------------
mlp    CONTROL. The static branch and the head only, with no sequence. It answers
       "how much of any gain comes from the sequence, and how much from using a
       neural network on the tabular features at all?"
lstm   static branch + LSTM over the 48-hour airport lookback + head.
gru    the same with a GRU.
transformer  the same with one small Transformer encoder block (DEC-022). The 3 channels
       are projected to ``units`` numbers per step, a LEARNED position embedding is
       added, then one block: 2-head self-attention (padded steps are masked out as
       keys), residual + LayerNorm, feed-forward (width ``dense``), residual +
       LayerNorm. The steps are averaged with the padding mask. There is no [CLS]
       token and no causal mask: every step is already in the past. Attention weights
       are not used as explanations.

Static branch: one embedding per categorical column (width from ``embedding_dim``),
concatenated with the ten numeric features, then Dense(dense) and Dropout.
Sequence branch: the recurrent layer (or the attention) gets the padding MASK, so padded
steps are skipped and cannot influence the output.
Head: concatenation, Dense(dense), Dropout, Dense(1, sigmoid).

The last layer's bias starts at the log-odds of the fit-row event rate, so training
starts from "predict the base rate" instead of spending epochs finding it.

Loss: binary cross-entropy, unweighted. Class imbalance is handled by calibration
and by ranking-based evaluation (scope), not by resampling or class weights here.
"""

from __future__ import annotations

import os
import time

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # keep the console readable

from airline_disruption.deep.batching import CATEGORICAL_INPUT_NAMES, BatchSource
from airline_disruption.deep.config import ARCHITECTURES, TRANSFORMER
from airline_disruption.deep.inputs import CATEGORICAL_COLUMNS, embedding_dim


def _keras():
    from tensorflow import keras

    return keras


def set_threads(n_threads: int) -> None:
    """Limit TensorFlow's CPU threads. Call before any model is built."""
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)


def _transformer_summary(sequence, mask, lookback: int, config: dict):
    """One encoder block over the lookback, then a masked mean: returns ``(batch, units)``.

    ``mask`` is True for real steps. Padded steps are removed twice: they are masked out as attention
    KEYS (so a real step never reads them) and they are excluded from the final average (so their
    own outputs never count). A row with no real step at all gets a summary of exactly zero, so it
    is scored from the static inputs alone, like the recurrent models.
    """
    keras = _keras()
    layers, ops = keras.layers, keras.ops
    width, heads = config["units"], TRANSFORMER["heads"]
    if width % heads != 0:
        raise ValueError(f"units={width} must be divisible by the number of attention heads ({heads})")

    keep = ops.cast(mask, "float32")  # (batch, lookback), 1.0 for a real step
    # Position index 0..lookback-1 built from the input itself, so the Embedding stays inside the model graph.
    # 0 = oldest step, lookback - 1 = newest. Padding always sits at the old end, so "newest" is always the same index.
    positions = ops.cast(ops.cumsum(ops.ones_like(keep), axis=1) - 1.0, "int32")

    hidden = layers.Dense(width, name="tf_project")(sequence)
    hidden = hidden + layers.Embedding(lookback, width, name="tf_position")(positions)

    # attention_mask[b, query, key] = 1 when the key is a real step. Shape (batch, lookback, lookback).
    key_ok = ops.cast(keep[:, None, :] * ops.ones_like(keep)[:, :, None], "bool")

    for block in range(TRANSFORMER["blocks"]):
        attended = layers.MultiHeadAttention(
            num_heads=heads, key_dim=width // heads, dropout=config["dropout"], name=f"tf_attention_{block}"
        )(hidden, hidden, attention_mask=key_ok)
        hidden = layers.LayerNormalization(name=f"tf_norm_a_{block}")(hidden + attended)
        feed = layers.Dense(config["dense"], activation="relu", name=f"tf_ff_in_{block}")(hidden)
        feed = layers.Dropout(config["dropout"], name=f"tf_ff_drop_{block}")(feed)
        feed = layers.Dense(width, name=f"tf_ff_out_{block}")(feed)
        hidden = layers.LayerNormalization(name=f"tf_norm_f_{block}")(hidden + feed)

    weights = keep[:, :, None]  # (batch, lookback, 1)
    total = ops.sum(hidden * weights, axis=1)
    count = ops.maximum(ops.sum(weights, axis=1), 1.0)  # avoids 0/0 for a row with no real step
    return total / count


def build_model(
    arch: str,
    vocabulary_sizes: dict[str, int],
    n_numeric: int,
    lookback: int,
    n_channels: int,
    config: dict,
    base_rate: float,
    seed: int,
):
    """Build and compile one model. ``vocabulary_sizes`` maps each categorical column to its row count."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be between 0 and 1")
    keras = _keras()
    layers = keras.layers
    keras.utils.set_random_seed(seed)

    inputs: dict[str, object] = {}
    embedded = []
    for input_name, column in zip(CATEGORICAL_INPUT_NAMES, CATEGORICAL_COLUMNS):
        size = vocabulary_sizes[column]
        inputs[input_name] = keras.Input(shape=(), dtype="int32", name=input_name)
        embedded.append(layers.Embedding(size, embedding_dim(size), name=f"embed_{input_name}")(inputs[input_name]))
    inputs["static_numeric"] = keras.Input(shape=(n_numeric,), dtype="float32", name="static_numeric")

    static = layers.Concatenate(name="static_concat")([*embedded, inputs["static_numeric"]])
    static = layers.Dense(config["dense"], activation="relu", name="static_dense")(static)
    static = layers.Dropout(config["dropout"], name="static_dropout")(static)

    if arch == "mlp":
        merged = static
    else:
        inputs["sequence"] = keras.Input(shape=(lookback, n_channels), dtype="float32", name="sequence")
        inputs["sequence_mask"] = keras.Input(shape=(lookback,), dtype="bool", name="sequence_mask")
        if arch == "transformer":
            summary = _transformer_summary(inputs["sequence"], inputs["sequence_mask"], lookback, config)
        else:
            recurrent = layers.LSTM if arch == "lstm" else layers.GRU
            summary = recurrent(config["units"], name=arch)(inputs["sequence"], mask=inputs["sequence_mask"])
        merged = layers.Concatenate(name="merge")([static, summary])

    head = layers.Dense(config["dense"], activation="relu", name="head_dense")(merged)
    head = layers.Dropout(config["dropout"], name="head_dropout")(head)
    log_odds = float(np.log(base_rate / (1.0 - base_rate)))
    output = layers.Dense(
        1, activation="sigmoid", bias_initializer=keras.initializers.Constant(log_odds), name="probability"
    )(head)

    model = keras.Model(inputs=inputs, outputs=output, name=f"phase6_{arch}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc", num_thresholds=1000)],
    )
    return model


def make_dataset(source: BatchSource, follow_shuffle: bool = True):
    """Wrap a BatchSource as a Keras ``PyDataset``.

    ``follow_shuffle=True`` (training): the order is ``source.order(epoch)`` and is reshuffled after
    every epoch when the source is shuffled. ``follow_shuffle=False`` (prediction and scoring): the
    rows are always visited in the source's row order, so scores line up with ``source.rows``.
    """
    keras = _keras()

    class _Batches(keras.utils.PyDataset):
        def __init__(self, src: BatchSource, follow: bool) -> None:
            super().__init__()
            self.src = src
            self.follow = follow
            self.epoch = 0
            self._order = self._make_order()

        def _make_order(self) -> np.ndarray:
            return self.src.order(self.epoch) if self.follow else np.arange(len(self.src.rows))

        def __len__(self) -> int:
            return len(self.src)

        def __getitem__(self, index: int):
            return self.src.get(index, self._order)

        def on_epoch_end(self) -> None:
            self.epoch += 1
            self._order = self._make_order()

    return _Batches(source, follow_shuffle)


def train_model(model, fit_source: BatchSource, stop_source: BatchSource, max_epochs: int, patience: int) -> dict:
    """Fit on ``fit_source``; stop early on the PR-AUC of ``stop_source``; restore the best weights.

    Only these two sources are passed in, so the validation year cannot influence training.
    Returns the per-epoch history, the best epoch (1-based), the epochs run and the seconds taken.
    """
    keras = _keras()
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True, verbose=0
    )
    started = time.perf_counter()
    history = model.fit(
        make_dataset(fit_source),
        validation_data=make_dataset(stop_source, follow_shuffle=False),
        epochs=max_epochs,
        callbacks=[stopper],
        verbose=2,
    )
    seconds = time.perf_counter() - started
    stop_curve = [float(v) for v in history.history["val_pr_auc"]]
    return {
        "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        "best_epoch": int(np.argmax(stop_curve)) + 1,
        "epochs_run": len(stop_curve),
        "best_stop_pr_auc": float(max(stop_curve)),
        "seconds": seconds,
    }


def predict_scores(model, source: BatchSource) -> np.ndarray:
    """Probabilities for ``source.rows`` in the source's unshuffled order."""
    scores = model.predict(make_dataset(source, follow_shuffle=False), verbose=0).reshape(-1).astype("float64")
    if len(scores) != len(source.rows):
        raise ValueError("The number of predictions does not match the number of rows")
    if not np.isfinite(scores).all():
        raise ValueError("The model produced non-finite probabilities")
    return scores


def count_parameters(model) -> int:
    return int(model.count_params())


def save_model(model, path) -> None:
    """Write the model (architecture and weights) to a ``.keras`` file."""
    model.save(str(path))


def load_model(path):
    """Read a model written by ``save_model``, for scoring only (it is not compiled, so it cannot be trained further)."""
    return _keras().models.load_model(str(path), compile=False)


def release() -> None:
    """Free the memory held by finished models (call after a model has been saved and scored)."""
    import gc

    _keras().backend.clear_session()
    gc.collect()
