import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

from pixelagent.anthropic import Agent

embed_model = sentence_transformer.using(model_id="all-mpnet-base-v2")

# First Create the Agent
agent = Agent(
    name="semantic_bot", system_prompt="You’re my assistant.", reset=True
)

# Add some memory
agent.chat("Hello my name is joe")

# Get the Agents Memory table and add embedding index to the content
memory = pxt.get_table("semantic_bot.memory")

memory.add_embedding_index(
    column="content",
    idx_name="content_idx",
    string_embed=embed_model,
    if_exists="replace",
)


def semantic_search(query: str) -> list[dict]:
    sim = memory.content.similarity(query, idx="content_idx")
    res = (
        memory.order_by(sim, asc=False)
        .select(memory.content, sim=sim)
        .limit(5)
        .collect()
    )
    result_str = "\n".join(
        f"Previous conversations: {content}" for content in res["content"]
    )
    return result_str


# Load some data into memory
agent.chat("I like football")

# test the semantic search
query = "I like denver broncos"
context_from_previous_conversations = semantic_search(query)
print(context_from_previous_conversations)
