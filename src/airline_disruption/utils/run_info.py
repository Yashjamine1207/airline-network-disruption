"""Facts about the machine and libraries behind a run.

Phase 6 requires every model run to record hardware context and library
versions. Models call these functions and store the result in their manifest.
"""

from __future__ import annotations

import os
import platform
import sys
from importlib import metadata


def package_versions(names: tuple[str, ...] = ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "tensorflow", "keras")) -> dict[str, str]:
    """Installed version of each package, or 'not installed'."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def hardware_context() -> dict[str, object]:
    """Python, operating system, CPU count and, if psutil is available, RAM."""
    info: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_total_gb": None,
    }
    try:
        import psutil

        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    return info
