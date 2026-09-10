"""Common pytest fixtures and configuration for pixelagent tests."""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Make tests/tools.py importable regardless of how pytest was invoked. The mock
# UDFs must live in an importable module: Pixeltable rejects a @pxt.udf defined in
# the global namespace of a script.
sys.path.insert(0, str(Path(__file__).parent))


def _ensure_pixeltable_home() -> tuple[str, bool]:
    """Point PIXELTABLE_HOME at a throwaway catalog unless the caller set one.

    Runs at conftest import time, before any test module imports pixeltable, so a
    developer's own ~/.pixeltable is never touched and every run starts empty.
    Returns (home, owned) where owned means this process created it.
    """
    existing = os.environ.get("PIXELTABLE_HOME")
    if existing:
        return existing, False
    home = os.path.join(tempfile.mkdtemp(prefix="pixelagent-pxt-"), "pixeltable")
    os.environ["PIXELTABLE_HOME"] = home
    return home, True


_PIXELTABLE_HOME, _OWNED = _ensure_pixeltable_home()


@pytest.fixture(scope="session", autouse=True)
def pixeltable_home():
    """Session-wide throwaway Pixeltable catalog."""
    yield _PIXELTABLE_HOME
    if _OWNED:
        shutil.rmtree(Path(_PIXELTABLE_HOME).parent, ignore_errors=True)


@pytest.fixture
def mock_stock_price():
    """A mock stock price tool returning a fixed integer."""
    from tools import stock_price_int

    return stock_price_int


@pytest.fixture
def mock_stock_price_dict():
    """A mock stock price tool returning a dictionary.

    Useful for OpenAI, which expects a richer return value.
    """
    from tools import stock_price_dict

    return stock_price_dict
