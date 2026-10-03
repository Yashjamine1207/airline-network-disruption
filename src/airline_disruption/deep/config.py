"""Pre-registered settings for the neural networks (no TensorFlow needed to import this).

Phase 6, Step 7. The defaults and the search space are fixed before any result exists
(DEC-020). The deep models follow the same select-on-2022-H1, confirm-on-2022-H2 rule as
LightGBM (DEC-019), with a much smaller budget because each fit is far slower. The number
of configurations tried is recorded beside every result.
"""

from __future__ import annotations

import numpy as np

ARCHITECTURES = ("mlp", "lstm", "gru", "transformer")

# The compact Transformer challenger (DEC-022). It reuses the searched sizes: ``units`` is the model
# width and ``dense`` is the feed-forward width. The rest is fixed, so it adds no new tuning freedom.
TRANSFORMER = {"heads": 2, "blocks": 1}

# The first configuration tried for every architecture.
DEFAULT_CONFIG = {"units": 32, "dense": 64, "dropout": 0.2, "learning_rate": 1e-3}

# Random draws come from here. Every list contains the default.
SEARCH_SPACE: dict[str, list] = {
    "units": [16, 32, 64],  # recurrent state size (ignored by the mlp control)
    "dense": [32, 64, 128],  # width of the static layer and of the head
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [1e-3, 3e-4],
}

# Same for every architecture and configuration.
TRAINING = {"batch_size": 2048, "max_epochs": 15, "patience": 3}


def sample_configs(n_draws: int, seed: int = 42) -> list[dict]:
    """``[default, draw 1, ..., draw n]``. Draws are distinct and never equal the default."""
    space_size = int(np.prod([len(v) for v in SEARCH_SPACE.values()]))
    if n_draws < 0:
        raise ValueError("n_draws must not be negative")
    if n_draws > space_size - 1:
        raise ValueError(f"n_draws={n_draws} is larger than the search space ({space_size - 1} other configurations)")
    rng = np.random.default_rng(seed)
    keys = list(SEARCH_SPACE)
    seen = {tuple(DEFAULT_CONFIG[k] for k in keys)}
    configs = [dict(DEFAULT_CONFIG)]
    while len(configs) < n_draws + 1:
        draw = {k: SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))] for k in keys}
        signature = tuple(draw[k] for k in keys)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(draw)
    return configs
