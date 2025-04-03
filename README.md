# Pixelagent: An Agent Engineering Blueprint 

We see agents as the intersection of an LLM, storage, and orchestration. Pixeltable unifies this interface into a single declarative framework, making it the de-facto choice for engineers to build custom agentic applications with build-your-own functionality for memory, tool-calling, and more.


## Build an Agent framework: 

- **Automated Data Orchestration**: Built on Pixeltable's infrastructure for seamless data management
- **Native Multimodal**: Built-in support for text, images, and beyond
- **Declarative Model**: Define tables and columns; Pixeltable handles the rest
- **LLM Protocol Support**: Handles OpenAI and Anthropic message protocols
- **Tool Integration**: Built-in tool-call handshake system
- **Agentic Extensions**: Add reasoning, reflection, memory, knowledge, and team workflows.

### Start with a simple Agent() class

- **[Build with Anthropic](examples/build-your-own-agent/single-provider/anthropic/README.md)**: Learn how we craft an agent using Claude, with cost-saving tricks like skipping chat history in tool calls.

- **[Build with OpenAI](examples/build-your-own-agent/single-provider/openai/README.md)**: See how we use GPT models to create a lean, powerful agent with the same Pixeltable-driven efficiency.

### Extend Agent() to multiple providers

- **[Build with Multiple Providers](examples/build-your-own-agent/multi-provider/README.md)**: Learn how to extend the Agent class to support multiple LLM providers.

## Plug-and-Play Extensions 

- **[Tools](examples/tool-calling)**: Add custom python functions as tools
- **[Memory](examples/memory)**: Implement long-term memory systems with semantic search capabilities
- **[Reflection](examples/reflection)**: Add self-improvement loops
- **[Reasoning](examples/planning)**: Add planning loops

## Usage

### Installation

```bash
pip install pixelagent
# Install provider-specific dependencies
pip install anthropic  # For Claude models
pip install openai     # For GPT models
```

### Quick Start

```python
from pixelagent.anthropic import Agent  # Or from pixelagent.openai import Agent

# Create a simple agent
agent = Agent(
    agent_name="my_assistant",
    system_prompt="You are a helpful assistant."
)

# Chat with your agent
response = agent.chat("Hello, who are you?")
print(response)
```

### Adding Tools

```python
import pixeltable as pxt
from pixelagent.anthropic import Agent
import yfinance as yf

# Define a tool as a UDF
@pxt.udf
def stock_price(ticker: str) -> dict:
    """Get stock information for a ticker symbol"""
    stock = yf.Ticker(ticker)
    return stock.info

# Create agent with tool
agent = Agent(
    agent_name="financial_assistant",
    system_prompt="You are a financial analyst assistant.",
    tools=pxt.tools(stock_price)
)

# Use tool calling
result = agent.tool_call("What's the current price of NVDA?")
print(result)
```

### Access Conversation Memory

```python
import pixeltable as pxt

# Agent memory is automatically persisted
memory = pxt.get_table("my_assistant.memory")
conversations = memory.collect()

# Access tool call history
tools_log = pxt.get_table("financial_assistant.tools")
tool_history = tools_log.collect()
```

### Advanced Features

```python
# Unlimited memory
infinite_agent = Agent(
    agent_name="historian",
    system_prompt="You remember everything.",
    n_latest_messages=None  # No limit on conversation history
)

# ReAct pattern for step-by-step reasoning and planning
import re
from datetime import datetime
from pixelagent.openai import Agent
import pixeltable as pxt

# Define a tool
@pxt.udf
def stock_info(ticker: str) -> dict:
    """Get stock information for analysis"""
    import yfinance as yf
    stock = yf.Ticker(ticker)
    return stock.info

# ReAct system prompt with structured reasoning pattern
REACT_PROMPT = """
Today is {date}

IMPORTANT: You have {max_steps} maximum steps. You are on step {step}.

Follow this EXACT step-by-step reasoning and action pattern:

1. THOUGHT: Think about what information you need to answer the question.
2. ACTION: Either use a tool OR write "FINAL" if you're ready to give your final answer.

Available tools:
{tools}

Always structure your response with these exact headings:

THOUGHT: [your reasoning]
ACTION: [tool_name] OR simply write "FINAL"
"""

# Helper function to extract sections from responses
def extract_section(text, section_name):
    pattern = rf'{section_name}:?\s*(.*?)(?=\n\s*(?:THOUGHT|ACTION):|$)'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Execute ReAct planning loop
def run_react_loop(question, max_steps=5):
    step = 1
    while step <= max_steps:
        # Dynamic system prompt with current step
        react_system_prompt = REACT_PROMPT.format(
            date=datetime.now().strftime("%Y-%m-%d"),
            tools=["stock_info"],
            step=step,
            max_steps=max_steps,
        )
        
        # Agent with updated system prompt
        agent = Agent(
            agent_name="financial_planner",
            system_prompt=react_system_prompt,
            reset=False,  # Maintain memory between steps
        )
        
        # Get agent's response for current step
        response = agent.chat(question)
        
        # Extract action to determine next step
        action = extract_section(response, "ACTION")
        
        # Check if agent is ready for final answer
        if "FINAL" in action.upper():
            break
            
        # Call tool if needed
        if "stock_info" in action.lower():
            tool_agent = Agent(
                agent_name="financial_planner",
                tools=pxt.tools(stock_info)
            )
            tool_agent.tool_call(question)
            
        step += 1
    
    # Generate final recommendation
    return Agent(agent_name="financial_planner").chat(question)

# Run the planning loop
recommendation = run_react_loop("Create an investment recommendation for AAPL")
```

Check out our [tutorials](examples/) for more examples including reflection loops, planning patterns, and multi-provider implementations.

## Tutorials and Examples

- **Basics**: Check out [Getting Started](examples/getting-started/pixelagent_basics_tutorial.py) for a step-by-step introduction to core concepts
- **Advanced Patterns**: Explore [Reflection](examples/reflection/anthropic/reflection.py) and [Planning](examples/planning/anthropic/react.py) for more complex agent architectures
- **Specialized Directories**: Browse our example directories for deeper implementations of specific techniques


Ready to start building? Dive into the blueprints, tweak them to your needs, and let Pixeltable handle the AI data infrastructure while you focus on innovation!
