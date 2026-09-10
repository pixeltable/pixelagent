
# prerequisite: run `python -m spacy download en_core_web_sm` first

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.openai import chat_completions
from pixeltable.functions.video import extract_audio
from pixeltable.functions.audio import audio_splitter
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.string import string_splitter

# Define the embedding model once for reuse
EMBED_MODEL = sentence_transformer.using(model_id='intfloat/e5-large-v2')

# Set up directory and table name
directory = 'video_index'
table_name = f'{directory}.video'

# Create video table
pxt.create_dir(directory, if_exists='replace_force')

video_index = pxt.create_table(
    table_name, 
    {'video': pxt.Video, 'uploaded_at': pxt.Timestamp}
)

video_index.add_computed_column(
    audio_extract=extract_audio(video_index.video, format='mp3')
) 

# Create view for frames
frames_view = pxt.create_view(
    f'{directory}.video_frames',
    video_index,
    iterator=frame_iterator(video=video_index.video, fps=1)
)

# Create a column for image description using OpenAI gpt-4o-mini
frames_view.add_computed_column(
    image_description=chat_completions(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Provide quick caption for the image."},
                {"type": "image_url", "image_url": {"url": frames_view.frame}},
            ],
        }],
    ).choices[0].message.content
)    

# Create embedding index for image description
frames_view.add_embedding_index('image_description', string_embed=EMBED_MODEL)    

# Create view for audio chunks
chunks_view = pxt.create_view(
    f'{directory}.video_chunks',
    video_index,
    iterator=audio_splitter(
        audio=video_index.audio_extract,
        duration=30.0,
        overlap=2.0,
        min_segment_duration=5.0,
    )
)

# Audio-to-text for chunks
chunks_view.add_computed_column(
    transcription=openai.transcriptions(
      audio=chunks_view.audio_segment, model='whisper-1'
    )
)

# Create view that chunks text into sentences
transcription_chunks = pxt.create_view(
    f'{directory}.video_sentence_chunks',
    chunks_view,
    iterator=string_splitter(text=chunks_view.transcription.text, separators='sentence'),
)

# Create embedding index for audio
transcription_chunks.add_embedding_index('text', string_embed=EMBED_MODEL)
