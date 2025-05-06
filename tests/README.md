# PixelAgent Tests

This directory contains tests for the PixelAgent library, focusing on testing the functionality of different model providers.

## Test Structure

The tests are organized by model provider:

- `test_anthropic.py`: Tests for the Anthropic provider
- `test_bedrock.py`: Tests for the AWS Bedrock provider
- `test_openai.py`: Tests for the OpenAI provider

Each provider has two standardized tests:
1. Basic chat functionality test
2. Tool calling functionality test

## Common Test Components

- `conftest.py`: Contains common fixtures used across all test files
- `pytest.ini`: Configuration for pytest

## Running Tests

### Running All Tests

```bash
pytest
```

### Running Tests for a Specific Provider

```bash
# Run only Anthropic tests
pytest -m anthropic

# Run only Bedrock tests
pytest -m bedrock

# Run only OpenAI tests
pytest -m openai
```

### Running Tests by Functionality

```bash
# Run only chat functionality tests
pytest -m chat

# Run only tool calling tests
pytest -m tool_calling
```

### Running a Specific Test File

```bash
pytest tests/test_anthropic.py
```

### Running with Verbose Output

```bash
pytest -v
```

## Test Markers

The following markers are available:

- `anthropic`: Tests for Anthropic provider
- `bedrock`: Tests for Bedrock provider
- `openai`: Tests for OpenAI provider
- `chat`: Tests for chat functionality
- `tool_calling`: Tests for tool calling functionality

## Mock Components

The tests use mock components to avoid external dependencies:

- `mock_stock_price`: A fixture that returns a fixed integer value (5)
- `mock_stock_price_dict`: A fixture that returns a dictionary with stock information

These mocks are used to test tool calling functionality without making actual API calls.
