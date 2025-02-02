from pxl.index import PixelIndex
from datetime import datetime

# Initialize the PDF index
index = PixelIndex(
    namespace="pixeltable_demo",
    table_name="financial_reports",
    index_type="pdf",
    clear_cache=True  # Set to False in production to preserve existing index
)

# Sample financial report PDFs
BASE_URL = "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/"
document_urls = [
    BASE_URL + doc for doc in [
        "Mclean-Equity-Alphabet.pdf",
        "Zacks-Nvidia-Repeport.pdf",
        # Add more documents as needed
    ]
]

# Ingest PDFs with metadata
for url in document_urls:
    # Extract filename for metadata
    filename = url.split("/")[-1]
    company = filename.split("-")[1]
    
    # Add relevant metadata for better filtering
    metadata = {
        "source": "financial_report",
        "company": company,
        "date_added": datetime.now(),
        "analyst": "McLean" if "Mclean" in filename else "Zacks",
        "document_type": "equity_research"
    }
    
    print(f"Ingesting: {filename}")
    index.insert(url, metadata=metadata)

# Example searches
print("\n1. Semantic Search Example:")
results = index.search(
    semantic_query="What are the key growth drivers for the company?",
    min_similarity=0.7,
    limit=3
)
print(results['text'])

print("\n2. Combined Semantic + Metadata Filter:")
results = index.search(
    semantic_query="What is the revenue forecast?",
    metadata_filters={"company": "Nvidia"},
    min_similarity=0.6,
    limit=2
)
print(results['text'])

print("\n3. Keyword Search with Metadata:")
results = index.search(
    keyword="Cash Flow From Operations",
    metadata_filters={"document_type": "equity_research"},
    limit=5
)
print(results['text'])

# Advanced example: Combining multiple metadata filters
print("\n4. Complex Search:")
results = index.search(
    semantic_query="What are the competitive advantages?",
    keyword="Cash Flow From Operations",
    metadata_filters={
        "company": "Alphabet",
        "analyst": "McLean"
    },
    min_similarity=0.75,
    limit=3
)
print(results['text'])
