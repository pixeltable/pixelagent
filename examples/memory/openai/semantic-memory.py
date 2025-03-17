# pip install sentence-transformers

from pixelagent.openai import Agent
import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer

embed_model = sentence_transformer.using(
    model_id="intfloat/e5-large-v2"
)

# First Create the Agent
agent = Agent(
    agent_name="long_term_bot",
    system_prompt="You’re my assistant.",
    reset=False
)

# Get the Agents Memory table and add embedding index to the content
memory = pxt.get_table("long_term_bot.memory")

@pxt.udf
async def concatenate_content(role: pxt.String, content: pxt.String, timestamp: pxt.Timestamp) -> str:
    return f"{str(timestamp)}: {role}: {content}"

memory.add_computed_column(
    user_content=concatenate_content(memory.role, memory.content, memory.timestamp),
    if_exists="ignore"
)

memory.add_embedding_index(column='user_content', idx_name="user_content_idx", string_embed=embed_model, if_exists="ignore")

def semantic_search(query: str) -> list[dict]:
    sim = memory.user_content.similarity(query, idx="user_content_idx")
    res = (memory.order_by(sim, asc=False)
            .select(memory.user_content, sim=sim)
            .limit(5)
            .collect())
    result_str = ""
    for i, row in enumerate(res.to_pandas().itertuples(), 1):
        result_str += f"Previous conversations: {row.user_content}\n"
    return result_str

# Load some data into memory
agent.chat("Hello my name is joe")
agent.chat("I like football")

# test the semantic search
query = "I like denver broncos"
context_from_previous_conversations = semantic_search(query) 
print(context_from_previous_conversations)