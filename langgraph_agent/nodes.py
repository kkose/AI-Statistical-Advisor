from rag.retriever import get_doc_answer
import os
from dotenv import load_dotenv
load_dotenv()

# LLM provider selection
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

# Always use Ollama if no OpenAI API key is set
if not OPENAI_API_KEY:
    from langchain_community.llms import Ollama
    llm_client = Ollama(model=OLLAMA_MODEL)
else:
    from openai import OpenAI
    llm_client = OpenAI()
import os
from dotenv import load_dotenv
load_dotenv()


## Use llm_client for completions

def classify_intent(state):
    """Check if the user question needs a doc search or can be answered directly."""
    question = state["question"]

    import re
    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that decides if a question about statistical tests needs document lookup or not. If it is about definitions or choosing the right test, return 'search'. Otherwise return 'simple'."},
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        decision = response.choices[0].message.content.strip().lower()
    except AttributeError:
        response = llm_client.invoke(
            f"You are an assistant that decides if a question about statistical tests needs document lookup or not. If it is about definitions or choosing the right test, return 'search'. Otherwise return 'simple'.\nQuestion: {question}"
        )
        decision = response.strip().lower()
    # Remove punctuation and whitespace
    decision = re.sub(r'[^a-z]', '', decision)
    # Only allow 'search' or 'simple'
    if 'search' in decision:
        decision = 'search'
    elif 'simple' in decision:
        decision = 'simple'
    else:
        decision = 'search'  # fallback
    print('*******')
    print(decision)
    print('*******')
    return {"intent": decision}
    

def retrieve_info(state):
    """Use the RAG tool to answer from embedded docs."""
    question = state["question"]
    answer = get_doc_answer(question=question)
    return {"rag_answer": answer}


def respond(state):
    """Build the final answer."""
    if state.get("rag_answer"):
        return {"final_answer": state["rag_answer"]}
    else:
        return {"final_answer": "I'm not sure how to help with that yet."}


def generate_code(state):
    """Generate Python code to perform the recommended statistical test."""
    question = state["question"]
    suggested_test = state.get("rag_answer") or "a statistical test"

    prompt = f"""
    You are a Python tutor. 
    Based on the following user question, generate a short Python code snippet using scipy.stats that performs the appropriate statistical test.

    User question:
    {question}

    Answer given:
    {suggested_test}

    Only output code. Don't include explanations.
    """

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"code_snippet": response.choices[0].message.content.strip()}
    except AttributeError:
        response = llm_client.invoke(prompt)
        return {"code_snippet": response.strip()}