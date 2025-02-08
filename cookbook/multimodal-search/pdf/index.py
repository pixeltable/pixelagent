import pixeltable as pxt
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators import DocumentSplitter

from pxl.agent import openai_agent

DIRECTORY = 'pdf_index'
TABLE_NAME = f'{DIRECTORY}.pdfs'
VIEW_NAME = f'{DIRECTORY}.pdf_chunks'
DELETE_INDEX = False

if DELETE_INDEX:
    pxt.drop_table(TABLE_NAME, force=True)

if TABLE_NAME not in pxt.list_tables():
    # Create documents table
    pxt.create_dir(DIRECTORY, if_exists='ignore')
    pdf_index = pxt.create_table(TABLE_NAME, {'pdf': pxt.Document})

    # Create view that chunks PDFs into sections
    chunks_view = pxt.create_view(
        VIEW_NAME,
        pdf_index,
        iterator=DocumentSplitter.create(document=pdf_index.pdf, separators='token_limit', limit=300),
    )

    # Define the embedding model
    embed_model = sentence_transformer.using(model_id='intfloat/e5-large-v2')

    # Create embedding index
    chunks_view.add_embedding_index(column='text', string_embed=embed_model)

else:
    pdf_index = pxt.get_table(TABLE_NAME)

# Sample PDFs
DOCUMENT_URL = 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/'
document_urls = [
    DOCUMENT_URL + doc
    for doc in [
        'Argus-Market-Digest-June-2024.pdf',
        'Argus-Market-Watch-June-2024.pdf',
        'Company-Research-Alphabet.pdf',
        'Jefferson-Amazon.pdf',
        'Mclean-Equity-Alphabet.pdf',
        'Zacks-Nvidia-Repeport.pdf',
    ]
]

# Add data to the table
pdf_index.insert({'pdf': url} for url in document_urls)

# Create search tool
@pxt.query
def search(query_text: str) -> str:
    """Search tool to find relevant passages.

    Args:
        query_text: The search query
    Returns:
        Top 10 most relevant passages
    """
    similarity = chunks_view.text.similarity(query_text)
    return (
        chunks_view.order_by(similarity, asc=False)
        .select(chunks_view.text, sim=similarity)
        .limit(10)
    )

# Create search agent
openai_agent.init(
    agent_name="PDF_Search",
    system_prompt="Use your tools to search the audio index",
    model_name="gpt-4o-mini",
    reset_memory=False,
    agent_tools=pxt.tools(search),
)


# Run the agent
result = openai_agent.run(
    agent_name="PDF_Search", message="Summarize the report on NVIDIA"
)

print(result)