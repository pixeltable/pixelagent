import importlib.util
import sys
import types
from pathlib import Path

import pytest

root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))


@pytest.fixture
def minimax_utils(monkeypatch):
    pxt = types.ModuleType("pixeltable")
    pxt.udf = lambda fn: fn
    monkeypatch.setitem(sys.modules, "pixeltable", pxt)
    utils_path = root_dir / "pixelagent" / "minimax" / "utils.py"
    spec = importlib.util.spec_from_file_location("minimax_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.minimax
def test_minimax_model_metadata(minimax_utils):
    assert set(minimax_utils.MINIMAX_MODELS) == {"MiniMax-M3", "MiniMax-M2.7"}
    assert minimax_utils.MINIMAX_MODELS["MiniMax-M3"]["context_window"] == 1_000_000
    assert minimax_utils.MINIMAX_MODELS["MiniMax-M3"]["input_modalities"] == [
        "text",
        "image",
        "video",
    ]
    assert minimax_utils.MINIMAX_MODELS["MiniMax-M2.7"]["context_window"] == 204_800
    assert minimax_utils.MINIMAX_MODELS["MiniMax-M2.7"]["thinking"] == ["always_on"]


@pytest.mark.minimax
def test_minimax_region_endpoints(minimax_utils):
    assert (
        minimax_utils.resolve_openai_base_url("global_en")
        == "https://api.minimax.io/v1"
    )
    assert (
        minimax_utils.resolve_openai_base_url("cn_zh")
        == "https://api.minimaxi.com/v1"
    )
    assert (
        minimax_utils.MINIMAX_ENDPOINTS["global_en"]["anthropic_base_url"]
        == "https://api.minimax.io/anthropic"
    )
    assert (
        minimax_utils.MINIMAX_ENDPOINTS["cn_zh"]["anthropic_base_url"]
        == "https://api.minimaxi.com/anthropic"
    )


@pytest.mark.minimax
def test_minimax_base_url_override_and_kwargs(minimax_utils):
    assert (
        minimax_utils.resolve_openai_base_url("cn_zh", "https://example.invalid/v1")
        == "https://example.invalid/v1"
    )

    kwargs = minimax_utils.with_base_url({"temperature": 0.2}, "global_en", None)

    assert kwargs == {
        "temperature": 0.2,
        "base_url": "https://api.minimax.io/v1",
    }


@pytest.mark.minimax
def test_minimax_invalid_region(minimax_utils):
    with pytest.raises(ValueError, match="Unsupported MiniMax region"):
        minimax_utils.resolve_openai_base_url("invalid")
