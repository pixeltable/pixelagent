class PixelagentError(Exception):
    """Base class for pixelagent errors."""


class SchemaConflict(PixelagentError):
    """
    Raised when `update_all()` refuses to reconcile an existing agent.

    `model` is a compile-time literal, so constructing the same name with a
    different model updates the computed column. `tools` is compiled into its
    own table. Pixeltable still rejects some of those edits (a toolset it cannot
    reconcile, a stored column whose type changed). This error is that refusal.

    Before 0.2.0 a repeat construction kept the old pipeline and discarded the
    new argument, so an agent could report a model it was not using.
    """

    def __init__(self, name: str, cause: Exception):
        self.agent_name = name
        self.cause = cause
        super().__init__(
            f"Agent {name!r} already exists, and Pixeltable refused this schema.\n"
            f"\n{cause}\n\n"
            f"A different `model` updates the chat pipeline in place; existing rows "
            f"keep their previous responses. Pass reset=True to rebuild {name!r} from "
            f"scratch (this drops its stored conversation history), or use a "
            f"different name.\n"
            f"system_prompt, model_kwargs, and max_tokens are stored on each turn."
        )
