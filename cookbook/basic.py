# pip install openai

from pxl.agent import Agent
from pxl.providers import Model
from pxl.index import PixelIndex

# from pxl.memory import Memory

# from pxl.index import Index
# from sentence_transformer import sentence_transformer

# embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# # Specialized index for advanced memory retrieval
# memory = PixelMemory(
#     index_name="memory",
#     type="temporal",
#     embedding_model=embed_model,
# )

# # Knowledge Base
# pdf = PixelIndex(
#     index_name="pdf_index",
#     index_type="document",
#     embedding_model=embed_model,
# )

# audio = PixelIndex(
#     index_name="audio_index",
#     index_type="audio",
#     embedding_model=embed_model,
# )

# video = PixelIndex(
#     index_name="video_index",
#     index_type="video",
#     embedding_model=embed_model,
# )

# Create Agent
llm = Model(provider="openai", model_name="gpt-4o-mini")

# ["autonomous", "semi-autonomous", "chain-of-thought", "tree-of-thought", "multi-agent"]
# pattern = "chain-of-thought" 
agent = Agent(
    model=llm,
    agent_name="Dog Trainer",
    system_prompt="You specialize in training dogs",
    # index=[pdf, audio, video],
    # memory=memory,
    # pattern=pattern,
    # clear_cache=True,
)

# Get answer
result = agent.run("in 5 words tell me how to train my dog to sit")
print(result)

# Inspect agent history
inspect = agent.get_history()
df = inspect.collect().to_pandas()
print(df.head())
