# 🤖 PixelAgent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Alpha-orange.svg" alt="Status">
</p>

**PixelAgent** is a powerful, lightweight framework for building AI agents with persistent memory, built on top of [Pixeltable](https://github.com/pixeltable/pixeltable). Create, deploy, and monitor sophisticated AI agents with just a few lines of code. ⚡ **Lightning fast** with **lowest level access** to the underlying models and data.

## ✨ Features

- ⚡ **Lightning Fast**: Optimized for speed and performance
- 🔧 **Low-Level API**: Direct control over model parameters and behavior
- 🧠 **Persistent Memory**: Every interaction is automatically stored in a Pixeltable database
- 🔌 **Multi-Model Support**: Works with OpenAI and Anthropic models out of the box
- 🛠️ **Tool Integration**: Easily add custom tools and functions to your agents
- 📊 **Structured Outputs**: Define Pydantic models for type-safe, structured responses
- 📝 **Conversation History**: Query and analyze full conversation history
- 🖼️ **Multimodal Support**: Handle text, images, and other media types
- 🔍 **Debugging Tools**: Built-in debugging and visualization capabilities

## 🚀 Installation

```bash
pip install pixelagent
```

## 🏁 Quick Start

### Basic Chat Agent

```python
from pixelagent.openai import Agent

agent = Agent(
    name="writer",
    system_prompt="You are a brilliant writer.",
    model="gpt-4o-mini",
    reset=True
)

result = agent.run("What is the capital of France?")
print(result)
```

### Agent with Custom Tools

```python
from pixelagent.openai import Agent, tool

@tool
def search_web(keywords: str, max_results: int) -> str:
    """Search the web for information."""
    # Simplified example
    results = [f"Result {i} for: {keywords}" for i in range(max_results)]
    return "\n".join(results)

agent = Agent(
    name="researcher",
    system_prompt="You are a research assistant that can search the web.",
    model="gpt-4o-mini",
    tools=[search_web]
)

response = agent.run("Find the latest news about AI")
print(response)
```

### Structured Output

```python
from typing import List
from pydantic import BaseModel
from pixelagent.openai import Agent, tool

class MovieRecommendation(BaseModel):
    title: str
    year: int
    genres: List[str]
    description: str
    rating: float

agent = Agent(
    name="movie_recommender",
    system_prompt="You recommend movies based on user preferences.",
    model="gpt-4o-mini",
    structured_output=MovieRecommendation,
    reset=True
)

movie = agent.run("Recommend a sci-fi movie from the 90s")
print(f"Title: {movie.title}, Year: {movie.year}, Rating: {movie.rating}")
```

### Multimodal Support

```python
from pixelagent.openai import Agent

image_url = "https://example.com/image.jpg"
agent = Agent(
    name="image_analyzer",
    system_prompt="You are an image analysis expert.",
    model="gpt-4o-mini",
    reset=True
)

response = agent.run("Analyze the image", attachments=image_url)
print(response)
```

## 📚 Documentation

For more examples and detailed documentation, check out the `cookbook` directory:

- **Basic Chat**: Simple conversation agents
- **Tool Calling**: Agents with custom tools and functions
- **Structured Outputs**: Type-safe responses with Pydantic
- **Multimodal**: Working with images and other media types

## 🧩 How It Works

PixelAgent uses Pixeltable as a persistent storage layer, automatically recording all conversations, tool calls, and agent states. This enables:

1. **Persistence**: Conversations continue where they left off
2. **Analysis**: Query your agent's history with SQL-like syntax
3. **Monitoring**: Track performance and behavior over time
4. **Debugging**: Identify and fix issues in your agent's reasoning

The framework is designed for **lightning-fast performance** with **lowest-level access** to model internals, giving you complete control while maintaining simplicity.

## 🔧 Known Issues

If you encounter a `KeyError: 'type'` when using tools, you need to update the tool definition format in `pixelagent/openai/utils.py`. The OpenAI API requires tools to have a specific format with a "type" field. Here's how to fix it:

```python
# In pixelagent/openai/utils.py, update the tool_dict definition:
tool_dict = {
    "type": "function",
    "function": {
        "name": func.__name__,
        "description": func.__doc__.strip() if func.__doc__ else f"Calls {func.__name__}",
        "parameters": parameters
    }
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

PixelAgent is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Pixeltable GitHub](https://github.com/pixeltable/pixeltable)
- [Twitter](https://twitter.com/pixeltableai)
- [Discord Community](https://discord.gg/pixeltable)
