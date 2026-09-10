"""Static gates on examples/.

These are greps and compiles rather than executions: the examples need media
files, provider keys and network access to run. They pin the classes of breakage
that made every example fail, so a regression shows up in CI instead of in a
user's terminal.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = sorted(ROOT.glob("examples/**/*.py"))


def test_examples_exist():
    assert EXAMPLES, "no examples found; the glob is wrong"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: str(p.relative_to(ROOT)))
def test_example_compiles(path):
    compile(path.read_text(), str(path), "exec")


def test_no_udf_in_a_script_global_namespace():
    """Pixeltable rejects a @pxt.udf defined in a script's global namespace.

    It must be importable by name so a stored computed column can find it again.
    Tool UDFs therefore live in a sibling *_tools.py module.
    """
    offenders = [
        str(p.relative_to(ROOT))
        for p in EXAMPLES
        if not p.name.endswith("tools.py") and re.search(r"^@pxt\.udf", p.read_text(), re.M)
    ]
    assert not offenders, (
        "@pxt.udf at script top level (move it into a module):\n" + "\n".join(offenders)
    )


def test_no_deprecated_iterator_api():
    """pixeltable.iterators shims forward kwargs verbatim with no name mapping,
    so a stale kwarg there is a hard error rather than a warning."""
    offenders = [
        f"{p.relative_to(ROOT)}: {m}"
        for p in EXAMPLES
        for m in re.findall(r"(?:from pixeltable\.iterators\S*|(?:Document|Audio|String)Splitter|FrameIterator)", p.read_text())
    ]
    assert not offenders, "deprecated iterator API:\n" + "\n".join(offenders)


def test_no_deprecated_openai_vision():
    offenders = [
        str(p.relative_to(ROOT))
        for p in EXAMPLES
        if re.search(r"from pixeltable\.functions\.openai import .*\bvision\b", p.read_text())
    ]
    assert not offenders, "openai.vision is deprecated; use chat_completions:\n" + "\n".join(offenders)


def test_no_stale_iterator_kwargs_or_columns():
    """The 0.4.x names: audio_splitter renamed these, and its output column too."""
    stale = ["chunk_duration_sec", "overlap_sec", "min_chunk_duration_sec", "html_skip_tags", "audio_chunk"]
    offenders = [
        f"{p.relative_to(ROOT)}: {name}"
        for p in EXAMPLES
        for name in stale
        if re.search(rf"\b{name}\b", p.read_text())
    ]
    assert not offenders, "stale iterator kwargs/columns:\n" + "\n".join(offenders)


def test_no_positional_similarity():
    offenders = [
        f"{p.relative_to(ROOT)}:{i}"
        for p in EXAMPLES
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if re.search(r"\.similarity\((?!string=|image=|vector=)", line)
    ]
    assert not offenders, "positional .similarity() is deprecated:\n" + "\n".join(offenders)
