"""Tests for airline_disruption.utils.run_info (Phase 6, Step 4)."""

from __future__ import annotations

from airline_disruption.utils.run_info import hardware_context, package_versions


def test_package_versions_reports_missing_packages() -> None:
    versions = package_versions(("pandas", "definitely_not_a_real_package_xyz"))
    assert versions["pandas"] != "not installed"
    assert versions["definitely_not_a_real_package_xyz"] == "not installed"


def test_hardware_context_has_the_basics() -> None:
    info = hardware_context()
    assert info["python"] and info["platform"]
    assert info["cpu_logical_cores"] >= 1
