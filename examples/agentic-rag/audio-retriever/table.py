
# prerequisite: run `python -m spacy download en_core_web_sm` first

import pixeltable as pxt
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter
import spacy

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

# Initialize app structure
pxt.drop_dir("audio_search", if_not_exists="ignore", force=True)
pxt.create_dir("audio_search")

# Create audio table
audio_t = pxt.create_table(
    "audio_search.audio", 
    {"audio_file": pxt.Audio}
)

# Add transcription workflow
audio_t.add_computed_column(
    transcription=whisper.transcribe(
        audio=audio_t.audio_file, 
        model="base.en"
    )
)

# Create sentence-level view
sentences_view = pxt.create_view(
    "audio_search.audio_sentence_chunks",
    audio_t,
    iterator=StringSplitter.create(
        text=audio_t.transcription.text, 
        separators="sentence"
    )
)

# Configure embedding model
embed_model = sentence_transformer.using(
    model_id="intfloat/e5-large-v2"
)

# Add search capability
sentences_view.add_embedding_index(
    column="text", 
    string_embed=embed_model
)
