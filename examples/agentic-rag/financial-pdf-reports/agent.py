import pixeltable as pxt
from pixelagent.openai import Agent

# Connect to your tables
documents_t = pxt.get_table("pdf_search.documents")
documents_chunks = pxt.get_table("pdf_search.document_chunks")

# Sample document URLs
DOCUMENT_URL = (
    "https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/"
)

document_urls = [
    DOCUMENT_URL + doc for doc in [
        "Argus-Market-Digest-June-2024.pdf",
        "Company-Research-Alphabet.pdf",
        # "Zacks-Nvidia-Report.pdf",
    ]
]

# Add documents to database
documents_t.insert({"pdf": url} for url in document_urls)

# Search documents
@pxt.query
def find_documents(query: str) -> dict:
    sim = documents_chunks.text.similarity(query)
    return (
        documents_chunks.order_by(sim, asc=False)
        .select(
            documents_chunks.text,
            similarity=sim
        )
        .limit(5)
    )

tools = pxt.tools(find_documents)
agent = Agent(
    name = "pdf_search.agent", 
    system_prompt = "Use your tool to search the PDF database.", 
    tools = tools
)

print(agent.tool_call("Use your tool to search for Alphabet earnings."))