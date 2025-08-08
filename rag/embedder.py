from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv
load_dotenv()


# Function to embed documents
def embed_docs(api_key):
    """
    Embeds the SciPy documentation using OpenAI embeddings and saves
    the result to a ChromaDB at "rag/chroma_db".

    This function is meant to be run once to generate the embeddings
    needed for the REPL.
    """
    # Load the documentation
    loader = TextLoader("docs/scipy_stats_docs.md", encoding= 'utf-8')
    docs = loader.load()

    # Split the documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
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

