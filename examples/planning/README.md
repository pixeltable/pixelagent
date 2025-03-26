# ReAct Pattern Tutorial

This tutorial demonstrates how to use PixelAgent's ReAct pattern to create agents that can reason about and execute complex multi-step tasks. We'll show how to build agents that effectively combine reasoning and action.

## What is ReAct?

ReAct (Reason+Act) is a pattern where agents alternate between:
- **Reasoning**: Thinking about what to do next and why
- **Acting**: Executing specific actions through tools
- **Observing**: Processing the results of actions

This creates a powerful loop that helps agents break down complex tasks into manageable steps and handle errors gracefully.

## Key Benefits

- **Structured Problem Solving**: Forces agents to think before they act
- **Better Error Handling**: Agents can reason about failures and try alternative approaches
- **Transparency**: Makes agent decision-making process visible and debuggable
- **Flexibility**: Works with any combination of tools and tasks

## How Pixeltable and Pixelagent Make ReAct Easy

Pixeltable and Pixelagent provide powerful infrastructure that makes implementing ReAct patterns seamless:

1. **Automatic Memory Management**: 
   - Persistent storage of agent conversations and tool outputs
   - Flexible memory windows (rolling or infinite)
   - No need to manually track conversation history

2. **Dynamic System Prompts**:
   - Easy injection of current state into prompts
   - Built-in support for step tracking and max steps
   - Automatic formatting of available tools

3. **Data Orchestration**:
   - Automatic storage and retrieval of tool outputs
   - Built-in support for parallel tool execution
   - Seamless handling of multimodal data

4. **Clean Implementation**:
   - No boilerplate code needed for memory or state management
   - Simple agent initialization with sensible defaults
   - Built-in support for common agent patterns

The framework handles all the complex data management and orchestration, letting you focus on designing your agent's reasoning flow. Check out the examples in this directory to see how easy it is to implement ReAct patterns with Pixelagent.

## How It Works

1. The agent receives a task
2. **Reason**: Agent thinks about what information or actions are needed
3. **Act**: Agent uses appropriate tools to gather info or make changes
4. **Observe**: Agent processes results and updates its understanding
5. Repeat until the task is complete

The ReAct pattern is particularly useful for tasks that require:
- Multiple steps to complete
- Dynamic decision making based on intermediate results
- Careful error handling and recovery
- Complex reasoning about tool usage

## Installation

Install Pixelagent and its dependencies:

```bash
pip install pixelagent yfinance openai anthropic
```

## Getting Started

You can find complete examples in both the `openai/` and `anthropic/` directories. Here's how to get started:

### OpenAI Example

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_api_key
   ```

2. Run the example:
   ```bash
   python openai/react.py
   ```

This will run a financial analysis agent that demonstrates the ReAct pattern by analyzing stock information step-by-step.

### Anthropic Example

1. Set your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=your_api_key
   ```

2. Run the example:
   ```bash
   python anthropic/react.py
   ```

Both examples show how to:
- Set up a ReAct agent with the appropriate system prompt
- Define and register tools using `@pxt.udf`
- Use Pixeltable's automatic memory management
- Handle the ReAct loop with step tracking

The examples use different underlying models (GPT-4 vs Claude) but follow the same pattern, demonstrating how Pixelagent makes it easy to swap between different LLM providers while maintaining the same ReAct structure.
