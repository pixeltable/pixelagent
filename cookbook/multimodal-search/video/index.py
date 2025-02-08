from datetime import datetime
import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.iterators.string import StringSplitter

from pxl.agent import openai_agent

DIRECTORY = 'video_index'
TABLE_NAME = f'{DIRECTORY}.video'
VIEW_NAME = f'{DIRECTORY}.video_chunks'
WHISPER_MODEL = 'whisper-1'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_dir(DIRECTORY, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create video table
    pxt.create_dir(DIRECTORY, if_exists='ignore')
    video_index = pxt.create_table(TABLE_NAME, {'video': pxt.Video, 'uploaded_at': pxt.Timestamp})

    # Video-to-audio
    video_index.add_computed_column(audio_extract=extract_audio(video_index.video, format='mp3'))

    # Audio-to-text
    video_index.add_computed_column(
        transcription=openai.transcriptions(audio=video_index.audio_extract, model=WHISPER_MODEL)
    )
    video_index.add_computed_column(transcription_text=video_index.transcription.text)

    # Create view that chunks text into sentences
    transcription_chunks = pxt.create_view(
        VIEW_NAME,
        video_index,
        iterator=StringSplitter.create(text=video_index.transcription_text, separators='sentence'),
    )

    # Define the embedding model
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    transcription_chunks.add_embedding_index('text', string_embed=embed_model)

else:
    video_index = pxt.get_table(TABLE_NAME)
    transcription_chunks = pxt.get_table(VIEW_NAME)

# Insert Videos
videos = [
    'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/audio-transcription-demo/'
    f'Lex-Fridman-Podcast-430-Excerpt-{n}.mp4'
    for n in range(3)
]

video_index.insert({'video': video, 'uploaded_at': datetime.now()} for video in videos[:2])

# Create search tool
@pxt.query
def search(query_text: str) -> str:
    """Search tool to find relevant passages.

    Args:
        query_text: The search query
    Returns:
        Top 10 most relevant passages
    """
    similarity = transcription_chunks.text.similarity(query_text)
    return (
        transcription_chunks.order_by(similarity, asc=False)
        .select(transcription_chunks.text, sim=similarity)
        .limit(10)
    )

# Create search agent
openai_agent.init(
    agent_name="Video_Search",
    system_prompt="Use your tools to search the video index",
    model_name="gpt-4o-mini",
    reset_memory=False,
    agent_tools=pxt.tools(search),
)


# Run the agent
result = openai_agent.run(
    agent_name="Video_Search", message="Explain what the author meant by what is happiness?"
)

print(result)