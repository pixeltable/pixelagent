import inspect
from functools import wraps
from typing import get_type_hints

def tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    for param_name, param in sig.parameters.items():
        param_type = type_hints.get(param_name, str)
        if param_type == str:
            schema_type = "string"
        elif param_type == int:
            schema_type = "integer"
        elif param_type == float:
            schema_type = "number"
        elif param_type == bool:
            schema_type = "boolean"
        else:
            schema_type = "string"

        param_schema = {"type": schema_type}
        if func.__doc__:
            param_schema["description"] = f"Parameter {param_name}"

        parameters["properties"][param_name] = param_schema
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    tool_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip()
            if func.__doc__
            else f"Calls {func.__name__}",
            "parameters": parameters,
            "strict": True,
        },
    }
    wrapper.tool_definition = tool_dict
    return wrapper