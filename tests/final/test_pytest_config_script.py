"""scripts/ensure_pytest_importlib.py edits pyproject.toml text safely and only once."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ensure_pytest_importlib", ROOT / "scripts/ensure_pytest_importlib.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OPTION = module.OPTION


def test_a_file_without_a_pytest_section_gets_one():
    new, message = module.add_importlib_mode('[project]\nname = "x"\n')
    assert new.endswith(f'[tool.pytest.ini_options]\naddopts = "{OPTION}"\n') and "new" in message


def test_an_existing_section_without_addopts_gets_the_line():
    new, _ = module.add_importlib_mode('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n\n[tool.ruff]\nline-length = 100\n')
    lines = new.splitlines()
    assert lines[1] == f'addopts = "{OPTION}"' and "[tool.ruff]" in new and 'testpaths = ["tests"]' in new


def test_an_existing_addopts_string_is_extended_not_replaced():
    new, _ = module.add_importlib_mode('[tool.pytest.ini_options]\naddopts = "-q"\n')
    assert f'addopts = "-q {OPTION}"' in new


def test_an_addopts_list_is_left_alone_with_a_message():
    text = '[tool.pytest.ini_options]\naddopts = ["-q"]\n'
    new, message = module.add_importlib_mode(text)
    assert new == text and "by hand" in message


def test_running_it_twice_changes_nothing_the_second_time():
    once, _ = module.add_importlib_mode('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    twice, message = module.add_importlib_mode(once)
    assert once == twice and "Nothing changed" in message


def test_an_addopts_in_another_section_is_not_touched():
    text = '[tool.other]\naddopts = "x"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    new, _ = module.add_importlib_mode(text)
    assert 'addopts = "x"' in new and f'addopts = "{OPTION}"' in new
