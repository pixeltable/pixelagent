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
        "additionalProperties": False,  # This is fine for strict mode
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
            schema_type = "string"  # Fallback

        param_schema = {"type": schema_type}
        if func.__doc__:
            param_schema["description"] = f"Parameter {param_name}"
        if param.default != inspect.Parameter.empty:
            param_schema["default"] = param.default
        else:
            parameters["required"].append(param_name)

        parameters["properties"][param_name] = param_schema

    # Updated tool_dict with strict=True
    tool_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.strip() if func.__doc__ else f"Calls {func.__name__}",
            "parameters": parameters,
            "strict": True
        }
    }
    wrapper.tool_definition = tool_dict
    return wrapper