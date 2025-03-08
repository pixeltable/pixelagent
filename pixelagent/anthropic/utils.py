from functools import wraps
from typing import Callable, Dict, Any

def _get_type_str(type_hint) -> str:
    """Convert Python type hints to JSON schema types."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array"
    }
    return type_map.get(type_hint, "string")

def tool(func: Callable) -> Callable:
    """Decorator to mark functions as tools."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    # Attach tool metadata to the function
    wrapper.name = func.__name__
    wrapper.description = func.__doc__.strip() if func.__doc__ else ""
    
    # Infer parameter type from type hint
    params = list(func.__annotations__.items())
    wrapper.parameters = {
        "type": "object",
        "properties": {
            name: {"type": _get_type_str(type_hint)}
            for name, type_hint in params if name != "return"
        },
        "required": [name for name, _ in params if name != "return"]
    }
    
    # Add to_dict method to the function
    def to_dict() -> Dict:
        """Convert tool to Anthropic API format."""
        return {
            "name": wrapper.name,
            "description": wrapper.description,
            "input_schema": wrapper.parameters
        }
    
    wrapper.to_dict = to_dict
    
    return wrapper