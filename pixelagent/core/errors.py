class PixelagentError(Exception):
    """Base class for pixelagent errors."""


class SchemaConflict(PixelagentError):
    """
    Raised when an agent is re-created under a name that already exists in the
    catalog with a different `model` or a different toolset.

    Pixeltable requires `model` to be a compile-time literal, and `invoke_tools`
    compiles the toolset into a column, so neither can vary per row. They are
    baked into the agent's schema, and changing one is a schema change.

    Before 0.2.0 this was silent: the existing pipeline was kept and the new
    argument was discarded, so an agent could report a model it was not using.
    """

    def __init__(self, name: str, cause: Exception):
        self.agent_name = name
        self.cause = cause
        super().__init__(
            f"Agent {name!r} already exists with a different model or toolset.\n"
            f"\n{cause}\n\n"
            f"`model` and `tools` are part of the agent's schema, so they cannot be "
            f"changed in place. Either pass reset=True to rebuild {name!r} from "
            f"scratch (this drops its stored conversation history), or use a "
            f"different name.\n"
            f"Config that DOES vary per call needs no rebuild: system_prompt, "
            f"model_kwargs, max_tokens and n_latest_messages are stored per row."
        )
