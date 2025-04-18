import pixeltable as pxt
from pixeltable.iterators import DocumentSplitter
from pixeltable.functions.huggingface import sentence_transformer

# Initialize app structure
pxt.drop_dir("pdf_search", if_not_exists="ignore", force=True)
pxt.create_dir("pdf_search")

# Create documents table
documents_t = pxt.create_table(
    "pdf_search.documents", 
    {"pdf": pxt.Document}
)

# Create chunked view for efficient processing
documents_chunks = pxt.create_view(
    "pdf_search.document_chunks",
    documents_t,
    iterator=DocumentSplitter.create(
        document=documents_t.pdf,
        separators="token_limit",
        limit=300  # Tokens per chunk
    )
)

# Configure embedding model
embed_model = sentence_transformer.using(
    model_id="intfloat/e5-large-v2"
)

# Add search capability
documents_chunks.add_embedding_index(
    column="text",
    string_embed=embed_model
)
