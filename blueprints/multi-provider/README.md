# Building Multi-Provider Agents with Pixeltable: A Step-by-Step Guide

This tutorial takes your agent engineering skills to the next level by showing you how to build a unified agent framework that works seamlessly across different LLM providers (Anthropic, OpenAI, and AWS Bedrock). We'll extend the single-provider `Agent()` class into a flexible architecture with a shared base class and provider-specific implementations.

## Prerequisites

- Install required packages:
  ```bash
  pip install pixeltable anthropic openai boto3
  ```
- Set up your API keys:
  ```bash
  export ANTHROPIC_API_KEY='your-anthropic-api-key'
  export OPENAI_API_KEY='your-openai-api-key'
  # For AWS Bedrock, configure AWS credentials using AWS CLI or environment variables
  aws configure  # Or set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION
  ```

## Introduction: Why Multi-Provider?

Building with multiple providers offers several advantages:

1. **Provider Flexibility**: Switch between models without changing your application code
2. **Failover Capability**: Fall back to alternative providers during outages
3. **Benchmark Different Models**: Compare performance across providers
4. **Future-Proofing**: Easily integrate new providers as they emerge
5. **Cost Optimization**: Choose providers based on price-performance ratio for different tasks

## The Architecture: From Single to Multi-Provider

The key to our multi-provider architecture is abstraction through inheritance:

1. **BaseAgent (Abstract Base Class)**: Contains all shared functionality including:
   - Table setup and management
   - Memory persistence logic
   - Core chat() and tool_call() implementations
   - Abstract methods for provider-specific logic

2. **Provider-Specific Agents**: Inherit from BaseAgent and implement only:
   - Provider-specific chat pipeline setup
   - Provider-specific tool handling
   - Default model configurations

This design pattern applies the principle of "code reuse through inheritance" while maintaining flexibility for provider-specific features.

## Step 1: Creating the Abstract Base Class

The heart of our multi-provider architecture is the `BaseAgent` abstract base classb:

```python
# multi-provider/core/base.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from uuid import uuid4

import pixeltable as pxt

class BaseAgent(ABC):
    """
    An Base agent powered by LLM model with persistent memory and tool execution capabilities.
    This base agent gets inherited by other agent classes.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        # Store configuration
        self.directory = name
        self.system_prompt = system_prompt
        self.model = model
        self.n_latest_messages = n_latest_messages
        self.tools = tools
        self.chat_kwargs = chat_kwargs or {}
        self.tool_kwargs = tool_kwargs or {}

        # Set up tables based on configuration
        if reset:
            pxt.drop_dir(self.directory, if_not_exists = "ignore", force=True)
        pxt.create_dir(self.directory, if_exists="ignore")
        self._setup_tables()
        
        # Get references to the created tables
        self.memory = pxt.get_table(f"{self.directory}.memory")
        self.agent = pxt.get_table(f"{self.directory}.agent")
        self.tools_table = pxt.get_table(f"{self.directory}.tools") if self.tools else None
```

The base class includes shared table setup and the primary user-facing methods (`chat()` and `tool_call()`) while declaring abstract methods that provider-specific implementations must define:

```python
    @abstractmethod
    def _setup_chat_pipeline(self):
        """To be implemented by subclasses"""
        raise NotImplementedError

    @abstractmethod
    def _setup_tools_pipeline(self):
        """To be implemented by subclasses"""
        raise NotImplementedError
```

## Step 2: Provider-Specific Implementations

Each provider implements its own agent class that inherits from BaseAgent. Let's see how the Anthropic implementation works:

```python
# multi-provider/anthropic.py

from typing import Optional

import pixeltable as pxt
import pixeltable.functions as pxtf

from pixelagent.core.base import BaseAgent

from .utils import create_messages


try:
    from pixeltable.functions.anthropic import invoke_tools, messages
except ImportError:
    raise ImportError("anthropic not found; run `pip install anthropic`")

class Agent(BaseAgent):
    """Anthropic-specific implementation of the BaseAgent."""
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "claude-3-5-sonnet-latest",  # Default model for Anthropic
        n_latest_messages: Optional[int] = 10,
        tools: Optional[pxt.tools] = None,
        reset: bool = False,
        chat_kwargs: Optional[dict] = None,
        tool_kwargs: Optional[dict] = None,
    ):
        # Initialize the base agent with all common parameters
        super().__init__(
            name=name,
            system_prompt=system_prompt,
            model=model,
            n_latest_messages=n_latest_messages,
            tools=tools,
            reset=reset,
            chat_kwargs=chat_kwargs,
            tool_kwargs=tool_kwargs,
        )
```

The Anthropic agent then implements the required abstract methods with Claude-specific logic:

```python
    def _setup_chat_pipeline(self):
        """Set up Anthropic-specific chat pipeline"""
        # Implementation details for Anthropic's message format
        # and API call structure...
        
    def _setup_tools_pipeline(self):
        """Set up Anthropic-specific tool execution pipeline"""
        # Implementation details for Anthropic's tool calling format...
```

Similarly, the OpenAI implementation follows the same pattern but with OpenAI-specific message formatting and API calls.

## Step 3: Using the Multi-Provider Architecture

With our architecture in place, using either provider becomes remarkably simple:

```python
# Import the specific provider you want to use
from .anthropic import Agent as AnthropicAgent
from .openai import Agent as OpenAIAgent
from .bedrock import Agent as BedrockAgent

# Create agents with the same interface
claude_agent = AnthropicAgent(
    name="claude_assistant",
    system_prompt="You are a helpful assistant.",
    model="claude-3-5-sonnet-latest"
)

gpt_agent = OpenAIAgent(
    name="gpt_assistant",
    system_prompt="You are a helpful assistant.",
    model="gpt-4-turbo"
)

bedrock_agent = BedrockAgent(
    name="bedrock_assistant",
    system_prompt="You are a helpful assistant.",
    model="amazon.nova-pro-v1:0"  # Or other Bedrock models
)

# Use them with the exact same interface
claude_response = claude_agent.chat("Tell me about quantum computing")
gpt_response = gpt_agent.chat("Tell me about quantum computing")
bedrock_response = bedrock_agent.chat("Tell me about quantum computing")
```

This unified interface makes it easy to:
- Swap providers with minimal code changes
- Run A/B tests between different models
- Create hybrid systems that use different providers for different tasks

## Advanced Features

### 1. Infinite Memory Support

All providers support infinite conversation history by setting `n_latest_messages=None`:

```python
# Create an agent with unlimited memory
agent = AnthropicAgent(
    name="memory_agent",
    system_prompt="You are a helpful assistant.",
    n_latest_messages=None  # No limit on conversation history
)
```

### 2. Tool Integration

All providers support the same tool interface:

```python
import pixeltable as pxt

@pxt.udf
def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

# Define tools
tools = pxt.tools(get_weather)

# Create agents with tools (works for all providers)
anthropic_agent = AnthropicAgent(
    name="anthropic_weather_assistant",
    system_prompt="Help users check weather.",
    tools=tools
)

openai_agent = OpenAIAgent(
    name="openai_weather_assistant",
    system_prompt="Help users check weather.",
    tools=tools
)

bedrock_agent = BedrockAgent(
    name="bedrock_weather_assistant",
    system_prompt="Help users check weather.",
    tools=tools
)
```

## Benefits of the Multi-Provider Architecture

The multi-provider architecture demonstrates the power of combining object-oriented programming principles with Pixeltable's declarative data orchestration. By abstracting common functionality into a base class and implementing provider-specific details in subclasses, we create a flexible, maintainable agent framework that works seamlessly across different LLM providers.

This approach not only reduces code duplication but also future-proofs your applications against changes in the AI landscape. As new models and providers emerge, you can simply add new implementations while maintaining the same consistent interface for your applications.

## Building on the Basics

- Explore creating your own provider implementations
- Build hybrid agents that leverage different models for different tasks
- Implement more advanced features like agent routing and model selection based on task complexity

## Advanced Patterns: Planning and Reflection

The multi-provider architecture provides an excellent foundation for implementing advanced agent patterns like planning and reflection. These patterns are particularly easy to build with our framework thanks to automatic memory management and dynamic system prompts.

### Planning with ReAct Pattern

The ReAct pattern (Reason + Act) enables agents to tackle complex multi-step tasks by alternating between reasoning and action:

```python
from .anthropic import Agent
import pixeltable as pxt

# Define financial tools
@pxt.udf
def stock_info(ticker: str) -> dict:
    """Get stock information"""
    # Implementation details...
    return data

# Create an agent with both system prompt and tools
agent = Agent(
    name="financial_planner",
    system_prompt="""
    You are a financial advisor that follows a step-by-step approach to analyze investments.
    
    Follow these steps when analyzing stocks:
    1. Think about what information you need
    2. Use the appropriate tools to gather that information
    3. Analyze the results before making decisions
    4. Explain your reasoning at each step
    
    Current step: {current_step} / {max_steps}
    """,
    tools=pxt.tools(stock_info)
)
```

**Why it's easy with our multi-provider architecture:**

1. **Automatic Memory Management**: All conversation history and tool outputs are automatically stored in Pixeltable, making it easy to reference previous steps
2. **Dynamic System Prompts**: Variables like `{current_step}` can be updated dynamically without rebuilding the agent
3. **Provider Flexibility**: The same ReAct pattern works seamlessly across Anthropic, OpenAI, and AWS Bedrock models

### Self-Reflection Pattern

The reflection pattern uses one agent to critique another agent's output, enabling continuous improvement:

```python
from .openai import Agent

# Main content generation agent
main_agent = Agent(
    name="content_creator",
    system_prompt="You are a technical writer creating clear documentation."
)

# Reflection agent for critique
reflection_agent = Agent(
    name="critic",
    system_prompt="You review technical documentation for clarity, accuracy, and completeness."
)

# Run reflection loop
def reflection_loop(user_query, iterations=2):
    # Generate initial response
    response = main_agent.chat(user_query)
    
    for i in range(iterations):
        # Get critique
        critique = reflection_agent.chat(f"Review this content:\n{response}")
        
        # Improve based on critique
        if "<OK>" not in critique:  # Simple check if improvement needed
            response = main_agent.chat(f"Improve your response based on this feedback:\n{critique}")
    
    return response
```

**Why it's easy with our multi-provider architecture:**

1. **Persistent Memory**: Both agents automatically maintain their conversation history, making it easy to analyze previous responses
2. **Uniform Interface**: The same code works with any combination of providers (OpenAI main + Anthropic reflection + Bedrock for specialized tasks, etc.)
3. **Simplified Agent Creation**: Creating specialized agents with different system prompts requires minimal code

### Cross-Provider Chains

One of the most powerful capabilities is creating agent chains that mix providers for different stages of processing:

```python
from .openai import Agent as OpenAIAgent
from .anthropic import Agent as AnthropicAgent
from .bedrock import Agent as BedrockAgent

# Use OpenAI for planning (strengths in structured reasoning)
planner = OpenAIAgent(
    name="task_planner",
    system_prompt="Break down complex tasks into steps."
)

# Use Anthropic for content generation (strengths in detailed explanations)
executor = AnthropicAgent(
    name="content_executor",
    system_prompt="Execute tasks according to plans."
)

# Use Bedrock for specialized tasks (e.g., code generation)
code_generator = BedrockAgent(
    name="code_generator",
    system_prompt="Generate code based on specifications.",
    model="amazon.nova-pro-v1:0"
)

# Multi-provider workflow
def execute_task(user_query):
    # Get plan from OpenAI agent
    plan = planner.chat(f"Create a plan for: {user_query}")
    
    # Execute plan with Anthropic agent
    result = executor.chat(f"Execute this plan:\n{plan}")
    
    # Generate code if needed
    if "code" in user_query.lower():
        code = code_generator.chat(f"Generate code for: {result}")
        result += f"\n\nHere's the implementation:\n\n{code}"
    
    return result
```

By leveraging our multi-provider architecture, you can build sophisticated agent systems that combine the strengths of different models while maintaining a clean, consistent interface. The automatic memory management and declarative approach make these advanced patterns remarkably easy to implement.

## Conclusion

Pixelagent's multi-provider architecture provides a powerful foundation for building advanced AI systems. Starting from a simple single-provider agent, we've shown how to create a flexible framework that supports multiple LLM providers, advanced patterns like planning and reflection, and sophisticated agent chains. 

The combination of object-oriented design principles and Pixeltable's declarative data orchestration creates an extensible system that can evolve with the rapidly changing AI landscape. By following the patterns in this tutorial, you can build agent systems that are not only powerful and flexible but also maintainable and future-proof.
