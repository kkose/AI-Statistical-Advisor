from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Always use Ollama embeddings if no OpenAI API key is set
if not OPENAI_API_KEY:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    EmbeddingsClass = OllamaEmbeddings
    ChatModelClass = Ollama
else:
    from langchain_openai import OpenAIEmbeddings
    from langchain_openai.chat_models import ChatOpenAI
    EmbeddingsClass = OpenAIEmbeddings
    ChatModelClass = ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

load_dotenv()


def get_doc_answer(question: str, k: int = 2) -> str:
    """
    Ask a question and get an answer from the vector store.

    Args:
        question: The question to ask
        k: The number of documents to retrieve

    Returns:
        The answer to the question
    """
    # Load vector store
    if not OPENAI_API_KEY:
        vectordb = Chroma(
            persist_directory="rag/chroma_db",
            embedding_function=EmbeddingsClass(model=OLLAMA_MODEL)
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        llm = ChatModelClass(model=OLLAMA_MODEL)
    else:
        vectordb = Chroma(
            persist_directory="rag/chroma_db",
            embedding_function=EmbeddingsClass()
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
        llm = ChatModelClass(model_name="gpt-4o", temperature=0)

    # Create a system prompt
    system_prompt= (
        "You are a helpful AI bot, specialist in statistical analysis."
        "You can answer questions using the given the SciPy documentation context."
        "Be clear and concise."
        "{context}"
    )

    # Create a prompt Template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create a chain
    # It creates a StuffDocumentsChain, which takes multiple documents (text data) and "stuffs" them together before passing them to the LLM for processing.
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    
    # Creates the RAG
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Invoke the chain
    result = retrieval_chain.invoke({"input": question})
    return result['answer']


if __name__ == "__main__":
    question = "What is the t-test used for?"
    answer = get_doc_answer(question)
    print("Answer:", answer)