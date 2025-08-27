from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", 'Ollama')
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

if PROVIDER == "Ollama":
    from langchain_ollama.embeddings import OllamaEmbeddings
    EmbeddingsClass = OllamaEmbeddings
else:
    from langchain_openai import OpenAIEmbeddings
    EmbeddingsClass = OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()


# Function to embed documents
def embed_docs():
    """
    Embeds the SciPy documentation using OpenAI embeddings and saves
    the result to a ChromaDB at "rag/chroma_db".

    This function is meant to be run once to generate the embeddings
    needed for the REPL.
    """
    # Load the documentation
    docs = []
    print("Embedding documents...")
    # Ingest markdown
    md_path = "docs/scipy_stats_docs.md"
    if os.path.exists(md_path):
        print(f"====>Loading {md_path}...")
        loader = TextLoader(md_path, encoding='utf-8')
        docs.extend(loader.load())
    # Ingest PDFs from docs/
    for fname in os.listdir("docs"):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join("docs", fname)
            pdf_loader = PyPDFLoader(pdf_path)
            docs.extend(pdf_loader.load())

    # Split the documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    if PROVIDER == "Ollama":
        embeddings = EmbeddingsClass(model=OLLAMA_MODEL)
    else:
        embeddings = EmbeddingsClass()
    
    # Save to ChromaDB
    vectordb = Chroma.from_documents(
        documents = split_docs,
        embedding = embeddings,
        persist_directory = "rag/chroma_db"
    )

    # Persist
    vectordb.persist()
    print("Documentation embedded and stored.")


if __name__ == "__main__":
    embed_docs()

