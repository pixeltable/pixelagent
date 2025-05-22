# Multi-Provider Agent Tests

This directory contains test scripts for the multi-provider agent blueprints. Each test demonstrates how to create and use agents powered by different LLM providers.

## Available Tests

- `test_anthropic.py`: Tests the Anthropic Claude agent with chat and tool calling
- `test_openai.py`: Tests the OpenAI GPT agent with chat and tool calling
- `test_bedrock.py`: Tests the AWS Bedrock agent with chat and tool calling

## Running the Tests

### Running Individual Tests

To run a specific test, navigate to the `blueprints/multi_provider` directory and run:

```bash
python -m tests.test_anthropic
```

Or for other providers:

```bash
python -m tests.test_openai
python -m tests.test_bedrock
```

### Running All Tests

To run all tests in sequence, use the provided runner script:

```bash
python -m tests.run_all_tests
```

You can also specify which tests to run:

```bash
python -m tests.run_all_tests test_anthropic test_openai
```

## Test Features

Each test demonstrates:

1. **Conversational Memory**: The agent remembers previous interactions within the conversation
2. **Tool Calling**: The agent can use tools to retrieve external information
3. **Memory Persistence**: The agent maintains memory across different interaction types

## Provider-Specific Notes

### Anthropic Claude

- Uses the `claude-3-5-sonnet-latest` model by default
- Demonstrates a weather tool

### OpenAI

- Uses the `gpt-4o-mini` model by default
- Demonstrates a weather tool

### AWS Bedrock

- Uses the `amazon.nova-pro-v1:0` model by default
- Demonstrates a stock price tool
- Requires AWS credentials to be configured

## Import Pattern

The tests demonstrate a simple import pattern:

```python
# Import the agent classes with aliases
from blueprints.multi_provider.ant.agent import Agent as AnthropicAgent
from blueprints.multi_provider.oai.agent import Agent as OpenAIAgent
from blueprints.multi_provider.bedrock.agent import Agent as BedrockAgent
```

This pattern makes it clear which provider's agent is being used while maintaining a consistent interface.

### For Application Code

When using these agents in your application code, you can use the same import pattern:

```python
from blueprints.multi_provider.ant.agent import Agent as AnthropicAgent
from blueprints.multi_provider.oai.agent import Agent as OpenAIAgent
from blueprints.multi_provider.bedrock.agent import Agent as BedrockAgent
```

The alias makes it clear which provider you're using, while the actual implementation uses directory names that avoid conflicts with the underlying Python packages.
