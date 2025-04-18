from datetime import datetime

import pixeltable as pxt
from pixelagent.openai import Agent

# Constants
directory = 'video_index'
table_name = f'{directory}.video'

# Connect to your tables and views
video_index = pxt.get_table(table_name)
transcription_chunks = pxt.get_table(f'{directory}.video_sentence_chunks')

# Insert videos to the knowledge base
videos = [
    'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/'
    f'Lex-Fridman-Podcast-430-Excerpt-{n}.mp4'
    for n in range(3)
]

video_index.insert({'video': video, 'uploaded_at': datetime.now()} for video in videos[:2])

@pxt.query
def search_transcription(query: str) -> dict:
    sim = transcription_chunks.text.similarity(query)
    return (
        transcription_chunks.order_by(sim, asc=False)
        .select(transcription_chunks.text)
        .limit(10)
    )

tools = pxt.tools(search_transcription)
agent = Agent(
    agent_name = "video_agent", 
    system_prompt = "Use your tool to search the video transcription.", 
    tools = tools,
    reset=True
)

print(agent.tool_call("Search the transcription for happiness. Summarize your findings"))
