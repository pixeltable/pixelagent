"""Keyless smoke tests: prove the Pixeltable pipeline is well formed.

Nothing here needs a provider credential. Each test either constructs an agent
(catalog work only) or inserts one row and inspects where the pipeline stopped.

The load-bearing assertion is `test_pipeline_reaches_the_provider`: every column
upstream of the API call computes cleanly, and the API call is the only thing that
fails. That separates "the schema is broken" from "there are no credentials",
which is what makes an unpinned-Pixeltable canary job useful.
"""

import datetime
import importlib
import os

import pixeltable as pxt
import pytest
from tools import stock_price_int

# provider label -> (module, model, name of the column feeding the API call, credential env var)
PROVIDERS = {
    "openai": ("pixelagent.openai", "gpt-4o-mini", "prompt", "OPENAI_API_KEY"),
    "anthropic": ("pixelagent.anthropic", "claude-3-7-sonnet-latest", "messages", "ANTHROPIC_API_KEY"),
    "bedrock": ("pixelagent.bedrock", "amazon.nova-pro-v1:0", "messages", "AWS_ACCESS_KEY_ID"),
    "gemini": ("pixelagent.gemini", "gemini-2.0-flash", "prompt", "GEMINI_API_KEY"),
}

ALL = list(PROVIDERS)


def insert_one_turn(agent) -> "object":
    """Write a single text-only turn straight into the agent table.

    Bypasses chat() so a failing API call is recorded on the row rather than
    raised, which is what lets a keyless run inspect where the pipeline stopped.
    """
    return agent.agent.insert(
        [
            {
                "message_id": "m1",
                "user_message": "hello",
                "timestamp": datetime.datetime.now(),
                "system_prompt": "You are a helpful assistant.",
                "image": None,
            }
        ],
        on_error="ignore",
    )


def make_agent(provider: str, name: str, **kwargs):
    module, model, _, _ = PROVIDERS[provider]
    agent_cls = importlib.import_module(module).Agent
    return agent_cls(
        name=name,
        system_prompt="You are a helpful assistant.",
        model=model,
        reset=True,
        **kwargs,
    )


@pytest.mark.parametrize("provider", ALL)
def test_agent_constructs(provider):
    """Every provider builds its directory and its two tables, with no credential."""
    agent = make_agent(provider, f"smoke_construct_{provider}")
    assert agent.agent is not None
    assert agent.memory is not None
    assert pxt.get_table(f"smoke_construct_{provider}.agent") is not None
    assert pxt.get_table(f"smoke_construct_{provider}.memory") is not None


@pytest.mark.parametrize("provider", ALL)
def test_agent_table_has_pipeline_columns(provider):
    """The chat pipeline is fully wired: history -> prompt -> response -> text."""
    _, _, prompt_col, _ = PROVIDERS[provider]
    agent = make_agent(provider, f"smoke_columns_{provider}")
    columns = agent.agent.columns()
    for expected in ("memory_context", prompt_col, "response", "agent_response"):
        assert expected in columns, f"{provider}: missing column {expected!r} in {columns}"


@pytest.mark.parametrize("provider", ALL)
def test_text_only_insert_accepts_a_null_image(provider):
    """Regression for the break that made every text-only chat() fail.

    Pixeltable 0.7.3 made declared types non-nullable, so `pxt.Image` rejected the
    `image=None` that chat() writes for a text-only turn. The column is now
    `pxt.Image | None`.
    """
    agent = make_agent(provider, f"smoke_null_image_{provider}")
    status = insert_one_turn(agent)
    assert status.num_rows == 1
    assert agent.agent.count() == 1


@pytest.mark.parametrize("provider", ALL)
def test_pipeline_reaches_the_provider(provider):
    """Everything upstream of the API call computes; only the API call fails.

    Run without credentials this is the strongest keyless assertion available: it
    fails if the schema, the query, the message formatting, or the kwarg routing
    breaks, and passes only when the pipeline got all the way to the network.
    """
    _, _, prompt_col, key = PROVIDERS[provider]
    if os.getenv(key):
        pytest.skip(f"{key} is set; this assertion is only meaningful without credentials")

    agent = make_agent(provider, f"smoke_reaches_{provider}")
    insert_one_turn(agent)

    t = agent.agent
    row = t.select(
        prompt=t[prompt_col],
        prompt_err=t[prompt_col].errortype,
        response_err=t.response.errortype,
        response_msg=t.response.errormsg,
    ).collect()

    assert row["prompt"][0] is not None, f"{provider}: {prompt_col} did not compute"
    assert row["prompt_err"][0] is None, f"{provider}: {prompt_col} errored: {row['prompt_err'][0]}"
    assert row["response_err"][0] is not None, (
        f"{provider}: expected the credential-less API call to fail, got a clean response"
    )


@pytest.mark.parametrize("provider", ALL)
def test_tools_pipeline_constructs(provider):
    """A tool-configured agent builds its tools table and full handshake pipeline."""
    agent = make_agent(provider, f"smoke_tools_{provider}", tools=pxt.tools(stock_price_int))
    assert agent.tools_table is not None
    columns = agent.tools_table.columns()
    for expected in ("initial_response", "tool_output", "tool_response_prompt", "final_response", "tool_answer"):
        assert expected in columns, f"{provider}: missing column {expected!r} in {columns}"


def test_anthropic_model_kwargs_compiles_into_a_computed_column():
    """The system prompt now rides in model_kwargs; the expression must compile.

    Whether Anthropic's SDK then receives it as a top-level `system` parameter can
    only be settled by a keyed call; see the `live` job.
    """
    from pixelagent.anthropic import Agent

    agent = Agent(
        name="smoke_anthropic_model_kwargs",
        system_prompt="You are a helpful assistant.",
        model="claude-3-7-sonnet-latest",
        max_tokens=1024,
        reset=True,
    )
    expr = repr(agent.agent.response.col.value_expr)
    assert "max_tokens=1024" in expr, expr
    assert "model_kwargs={'system': 'You are a helpful assistant.'}" in expr, expr
    assert "system=" not in expr.replace("'system'", ""), expr


@pytest.mark.parametrize("provider", ALL)
def test_reconstructing_by_name_reattaches(provider):
    """Constructing the same agent again reuses the existing tables."""
    module, model, _, _ = PROVIDERS[provider]
    agent_cls = importlib.import_module(module).Agent
    name = f"smoke_reattach_{provider}"

    first = agent_cls(name=name, system_prompt="You are a helpful assistant.", model=model, reset=True)
    first_columns = first.agent.columns()

    second = agent_cls(name=name, system_prompt="You are a helpful assistant.", model=model)
    assert second.agent.columns() == first_columns
