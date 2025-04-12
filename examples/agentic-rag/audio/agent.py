import pixeltable as pxt
from pixelagent.openai import Agent

# Connect to your tables and views
audio_t = pxt.get_table("audio_search.audio")
sentences_view = pxt.get_table("audio_search.audio_sentence_chunks")

# Add audio files to the knowledge base
audio_t.insert([{
    "audio_file": "https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/10-minute%20tour%20of%20Pixeltable.mp3"
}])

# Perform search
@pxt.query()
def audio_search(query_text: str, n: int = 5):
    min_similarity = 0.8
    sim = sentences_view.text.similarity(query_text)
    return (
        sentences_view.where(sim >= min_similarity)
        .order_by(sim, asc=False)
        .select(sentences_view.text, sim=sim)
        .limit(n)
    )

agent = Agent(
    name = "audio_search.agent", 
    system_prompt = "Use your tool to search the audio database.", 
    tools = pxt.tools(audio_search)
)

agent.tool_call("search for Pixeltable best practices")