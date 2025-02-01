# Pixel Agent

Build-your-own agent framework with Pixeltable.

## Usage

```python
from pxl.agent import Agent
from pxl.providers import Model

# Create Agent
llm = Model(provider="openai", model_name="gpt-4o-mini")
agent = Agent(
    model=llm,
    agent_name="Dog Trainer",
    system_prompt="You specialize in training dogs",
    # clear_cache=True,
)


# Get answer
result = agent.run("in 5 words tell me how to train my dog to sit")
print(result)

# Inspect agent history
inspect = agent.get_history()
df = inspect.collect().to_pandas()
print(df.head())
```
