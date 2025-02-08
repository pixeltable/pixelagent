from pxl.index import MultimodalIndex
from datetime import datetime
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

# Create a simple audio index with default settings
index = MultimodalIndex(
    namespace="podcast_library",     # Where to store the index
    table_name="episodes",           # What to call our table
    index_types=["audio"]            # Type of content we're indexing
)

# For advanced users who want to customize their processing:
advanced_index = MultimodalIndex(
    namespace="custom_podcast_library",
    table_name="episodes",
    index_types=["audio", "document"],
    # Define Whisper model
    model=lambda audio: whisper.transcribe(
        audio=audio, 
        model="base.en"
    ),
    # Define chunking strategy
    chunking=lambda text: StringSplitter.create(
        text=text,
        separators="sentence"  # Split by paragraphs instead of sentences
    ),
    # Define embedding model 
    embeddings=sentence_transformer.using(
        model_id="sentence-transformers/all-mpnet-base-v2"
    )
)

# Sample podcast episodes to index
podcasts = [
    {
        "file": "s3://pixeltable-public/audio/pixeltable-intro.mp3",
        "metadata": {
            "title": "Introduction to Pixeltable",
            "host": "Sarah Chen",
            "guests": ["Alex Kumar"],
            "recorded": datetime(2024, 3, 15),
            "duration_mins": 15,
            "topics": ["data science", "AI"]
        }
    }
]

# Add podcasts to our index
for podcast in podcasts:
    print(f"📝 Indexing: {podcast['metadata']['title']}")
    index.insert(podcast["file"], metadata=podcast["metadata"])
