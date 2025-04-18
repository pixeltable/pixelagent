# pip install sentence-transformers

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions.openai import embeddings
from pixelagent.openai import Agent

embed_model = embeddings.using(model="text-embedding-3-small")

# First Create the Agent
agent = Agent(
    agent_name="conversational_agent", 
    system_prompt="You’re my assistant.", 
    reset=True,
    n_latest_messages=10
)

# Get the Agents Memory table and add embedding index to the content
# This contains conversation history
memory = pxt.get_table("rolling_memory_agent.memory")

memory.add_computed_column(
    user_content=pxtf.string.format(
        "{0}: {1}", memory.role, memory.content
    ),
    if_exists="ignore",
)

memory.add_embedding_index(
    column="user_content",
    idx_name="user_content_idx",
    embedding=embed_model,
    if_exists="ignore",
)

# We can search for context from previous conversations that semantically match the query
@pxt.query
def search_memory(query: str) -> list[dict]:
    sim = memory.user_content.similarity(query, idx="user_content_idx")
    res = (
        memory.order_by(sim, asc=False)
        .select(memory.user_content, sim=sim)
        .limit(5)
    )
    return res

memory_agent = Agent(
    agent_name="semantic_memory_agent",
    system_prompt="Fetch context from previous conversations.",
    tools = pxt.tools(search_memory)
)

# Load some data into memory
print(agent.chat("Can you recommend some activities in Ireland?"))

print("--------------------------------------------\n")
print(agent.chat("Can you recommend some activities in Paris?"))

print("--------------------------------------------\n")
print(memory_agent.tool_call("search and summarize memories about ireland"))

