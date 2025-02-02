from pxl.index import PixelIndex
from datetime import datetime

# Initialize the Audio index
index = PixelIndex(
    namespace="pixeltable_demo",
    table_name="podcast_library",
    index_type="audio",
    clear_cache=True  # Set to False in production to preserve existing index
)

# Sample podcast episodes
BASE_URL = "s3://pixeltable-public/audio/"
audio_files = [
    {
        "url": BASE_URL + "10-minute tour of Pixeltable.mp3",
        "metadata": {
            "show_name": "Tech Talks",
            "episode_title": "Introduction to Pixeltable",
            "speaker": "John Doe",
            "recorded_at": datetime(2024, 1, 15),
            "duration_minutes": 10.5,
            "topics": ["data science", "AI", "technology"],
            "language": "english",
            "episode_type": "tutorial"
        }
    }
]

# Ingest audio files with metadata
for audio in audio_files:
    print(f"Ingesting: {audio['metadata']['episode_title']}")
    index.insert(audio["url"], metadata=audio["metadata"])

# Example searches showcasing different capabilities
print("\n1. Semantic Search Example:")
results = index.search(
    semantic_query="Explain how Pixeltable works with databases",
    min_similarity=0.7,
    limit=3
)
print(results)

print("\n2. Speaker-Specific Search:")
results = index.search(
    semantic_query="What are the key benefits of AI?",
    metadata_filters={"speaker": "Jane Smith"},
    min_similarity=0.6,
    limit=2
)
print(results)

print("\n3. Topic-Based Search:")
results = index.search(
    keyword="machine learning",
    metadata_filters={"episode_type": "interview"},
    limit=3
)
print(results)

print("\n4. Complex Search:")
results = index.search(
    semantic_query="What are the ethical considerations in AI?",
    metadata_filters={
        "show_name": "Future Tech",
        "topics": ["AI", "ethics"]
    },
    min_similarity=0.75,
    limit=2
)
print(results)

# Time-based search example
print("\n5. Recent Content Search:")
results = index.search(
    semantic_query="Latest developments in AI",
    metadata_filters={
        "recorded_at": datetime(2024, 2, 1),
        "duration_minutes": {"$lt": 30}  # Episodes shorter than 30 minutes
    },
    min_similarity=0.6,
    limit=2
)
print(results)
