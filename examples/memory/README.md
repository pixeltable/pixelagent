# Agent Memory Examples

Pixelagent's memory system demonstrates the power of our data-first approach to agent development. By leveraging Pixeltable's AI data infrastructure, you get a production-ready memory system with zero setup - just focus on building your agent's logic.

## Why Use Pixelagent for Memory?

- **Data-First**: Memory is automatically persisted and queryable through Pixeltable
- **Engineering Freedom**: Simple interface that doesn't constrain your implementation
- **Simplified Workflow**: Automated handling of:
  - Memory persistence and retrieval
  - Semantic search capabilities
  - State management
  - Timestamp tracking

## Basic Memory Access

The simplest way to access agent memory is through the Pixeltable table interface. See [basic.py](openai/basic.py):

```bash
pip install -r examples/memory/openai/requirements.txt
```

```python
import pixeltable as pxt
from pixelagent.openai import Agent

# Create an agent
agent = Agent(name="openai_agent", system_prompt="You're my assistant.", reset=True)

# Chat with the agent
agent.chat("Hi, how are you?")
agent.chat("What was my last question?")

# Access memory through Pixeltable
memory = pxt.get_table("openai_agent.memory")
print(memory.collect())  # Shows all conversations
```

The memory table contains columns like:
- `timestamp`: When the message was sent/received
- `role`: Who sent the message (user/assistant)
- `content`: The actual message content

## Semantic Memory Search

For more advanced use cases, you can add semantic search capabilities to find relevant past conversations. See [semantic-memory.py](openai/semantic-memory.py):

```python
from pixelagent.openai import Agent
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.huggingface import sentence_transformer

# Setup embedding model
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Create agent
agent = Agent(name="semantic_bot", system_prompt="You're my assistant.", reset=False)

# Get memory table and add embedding index
memory = pxt.get_table("semantic_bot.memory")
memory.add_computed_column(
    user_content=pxtf.string.format("{0}: {1}: {2}", 
                                  memory.timestamp, memory.role, memory.content),
    if_exists="ignore"
)
memory.add_embedding_index(
    column="user_content",
    idx_name="user_content_idx",
    string_embed=embed_model,
    if_exists="ignore"
)

# Function to search memory semantically
def semantic_search(query: str) -> str:
    sim = memory.user_content.similarity(query, idx="user_content_idx")
    res = (memory.order_by(sim, asc=False)
           .select(memory.user_content, sim=sim)
           .limit(5)
           .collect())
    return "\n".join(f"Previous conversations: {user_content}" 
                    for user_content in res["user_content"])
```

## Key Features

1. **Persistent Storage**: All conversations are automatically stored in Pixeltable
2. **Easy Access**: Simple interface to query historical conversations
3. **Semantic Search**: Optional semantic search capability using embeddings
4. **Timestamp-based**: All memories are timestamped for chronological access
5. **Flexible Querying**: Full SQL-like query capabilities through Pixeltable

## Installation

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python -m venv .venv
source .venv/bin/activate
```

2. Install basic requirements:
```bash
# For OpenAI example
pip install pixeltable pixelagent openai

# For Anthropic example
pip install pixeltable pixelagent anthropic
```

3. For semantic memory (optional):
```bash
pip install sentence-transformers spacy
python -m spacy download en_core_web_sm
```

## Requirements

### Basic Memory Example
- pixeltable
- pixelagent
- openai (for OpenAI example) or anthropic (for Anthropic example)

### Semantic Memory Example
Additional requirements for semantic search:
- sentence-transformers
- spacy
