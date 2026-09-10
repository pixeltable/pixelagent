"""
The agent's schema, declared rather than assembled.

Before 0.2.0 an agent was built by a sequence of catalog mutations: create the
tables, then `add_computed_column(..., if_exists="ignore")` for each stage. That
has no notion of drift. Re-running it against an existing agent silently kept
whatever was there, so an agent could report a model it was not using.

Here the tables are declared as `TableModel` classes and reconciled by
`update_all()`, which diffs them against the catalog and fails loudly on a
conflict.

The one thing that makes this possible for a library -- as opposed to an
`app.py`, where the schema is fixed at authoring time -- is that
`pxt.model_base()` returns a *fresh* model registry on every call. So each agent
gets its own registry, and `model` and `tools` can be closed over as literals
while every other setting stays row data.
"""

from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from .errors import SchemaConflict
from .spec import ProviderSpec

# `n_latest_messages=None` means "the whole conversation". A query needs a real
# limit, so None maps to a bound no conversation will reach.
UNLIMITED_HISTORY = 1_000_000


def build(
    dir_name: str,
    spec: ProviderSpec,
    *,
    model: str,
    tools: Optional[pxt.Tools] = None,
    reset: bool = False,
):
    """
    Declare and reconcile one agent's tables. Returns the model base, whose
    tables live under `dir_name`.

    Raises SchemaConflict if `dir_name` already holds an agent built with a
    different model or toolset.
    """
    if reset:
        pxt.drop_dir(dir_name, force=True, if_not_exists="ignore")
    pxt.create_dir(dir_name, if_exists="ignore")

    TM = pxt.model_base()

    class Memory(TM, name="memory"):
        """Conversation history. One row per message, user and assistant alike."""

        conversation_id: pxt.String
        role: pxt.String
        content: pxt.String
        timestamp: pxt.Timestamp
        uuid = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)
        # 0.7.2 stopped creating b-tree indexes by default, and every turn
        # filters and orders on these two.
        __indexes__ = [pxt.BtreeIndex(conversation_id), pxt.BtreeIndex(timestamp)]

    @pxt.query
    def history(conversation_id: str, before: pxt.Timestamp, n: int):
        """
        The n most recent messages in a conversation, strictly before `before`.

        Strictly before matters: chat() stamps the agent row and the user's
        memory row with the same instant, so the current turn is excluded from
        its own context here and re-appended by build_prompt. That was the
        behaviour of the hand-written pipeline and it is preserved deliberately.
        """
        return (
            Memory.where(
                (Memory.conversation_id == conversation_id)
                & (Memory.timestamp < before)
            )
            .order_by(Memory.timestamp, asc=False)
            .select(role=Memory.role, content=Memory.content)
            .limit(n)
        )

    class Agent(TM, name="agent"):
        """
        One row per turn. Inserting a row runs the whole pipeline.

        Everything here except the model and the toolset is per-row data, which
        is what lets one schema serve an agent whose system prompt or decoding
        settings change from call to call.
        """

        conversation_id: pxt.String
        user_message: pxt.String
        system_prompt: pxt.String
        model_kwargs: pxt.Json | None
        max_tokens: pxt.Int
        n_latest: pxt.Int
        image: pxt.Image | None
        timestamp: pxt.Timestamp
        uuid = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)

        memory_context = history(conversation_id, timestamp, n_latest)
        prompt = spec.build_prompt(system_prompt, memory_context, user_message, image)
        response = spec.call(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            model_kwargs=model_kwargs,
            max_tokens=max_tokens,
            tools=None,
        )
        agent_response = spec.extract(response)

    if tools is not None:
        class Tools(TM, name="tools"):
            """
            The tool-calling handshake: ask with tools advertised, run whatever
            the model picked, then ask again with the results.
            """

            tool_prompt: pxt.String
            system_prompt: pxt.String
            model_kwargs: pxt.Json | None
            max_tokens: pxt.Int
            timestamp: pxt.Timestamp
            uuid = pxt.Column(value=pxtf.uuid.uuid7(), primary_key=True)

            initial_response = spec.call(
                model=model,
                prompt=spec.wrap_user_text(tool_prompt),
                system_prompt=system_prompt,
                model_kwargs=model_kwargs,
                max_tokens=max_tokens,
                tools=tools,
            )
            tool_output = spec.invoke(tools, initial_response)
            tool_response_prompt = pxtf.string.format(
                "{0}: {1}", tool_prompt, tool_output
            )
            final_response = spec.call(
                model=model,
                prompt=spec.wrap_user_text(tool_response_prompt),
                system_prompt=system_prompt,
                model_kwargs=model_kwargs,
                max_tokens=max_tokens,
                tools=None,
            )
            tool_answer = spec.extract(final_response)

    try:
        TM.update_all(dir_name)
    except pxt.Error as e:
        raise SchemaConflict(dir_name, e) from e

    return TM
