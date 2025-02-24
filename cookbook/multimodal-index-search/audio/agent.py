import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

from pixelagent.openai import Agent, tool

DIRECTORY = "audio_index"
TABLE_NAME = f"{DIRECTORY}.audio"
VIEW_NAME = f"{DIRECTORY}.audio_sentence_chunks"

# Create audio index
pxt.create_dir(DIRECTORY, if_exists="ignore")
audio_index = pxt.create_table(
    TABLE_NAME, {"audio_file": pxt.Audio}, if_exists="ignore"
)
audio_index.add_computed_column(
    transcription=whisper.transcribe(audio=audio_index.audio_file, model="base.en"),
    if_exists="ignore",
)
sentences_view = pxt.create_view(
    VIEW_NAME,
    audio_index,
    iterator=StringSplitter.create(
        text=audio_index.transcription.text, separators="sentence"
    ),
    if_exists="ignore",
)
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")
sentences_view.add_embedding_index(
    column="text", string_embed=embed_model, if_exists="ignore"
)

# Add data to the table
audio_index.insert(
    [{"audio_file": "s3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3"}]
)


similarity = sentences_view.text.similarity("what is pixeltable?")
print(
    sentences_view.order_by(similarity, asc=False)
    .select(sentences_view.text, sim=similarity)
    .limit(10)
)
