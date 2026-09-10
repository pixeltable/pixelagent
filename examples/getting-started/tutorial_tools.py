"""Tool UDFs for pixelagent_basics_tutorial.py.

They live in a module rather than beside the agent because Pixeltable refuses a
@pxt.udf defined in the global namespace of a script: a UDF has to be importable
by name so a stored computed column can find it again on a later run.
"""


import pixeltable as pxt


@pxt.udf
def weather(city: str) -> str:
    """
    Get the current weather for a specified city.

    Args:
        city (str): The name of the city to check weather for

    Returns:
        str: Weather description for the requested city
    """
    return f"The weather in {city} is sunny."
