from rag.retriever import get_doc_answer
import os
from dotenv import load_dotenv
load_dotenv()

# LLM provider selection
PROVIDER = os.environ.get("LLM_PROVIDER", None)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

if PROVIDER == "OpenAI" and not OPENAI_API_KEY:
    PROVIDER = "Ollama"

if PROVIDER == "Ollama":
    from langchain_ollama import Ollama
    llm_client = Ollama(model=OLLAMA_MODEL)
else:
    from openai import OpenAI
    llm_client = OpenAI()

def classify_intent(state):
    question = state["question"]

    if PROVIDER == "Ollama":
        system_prompt = (
            "You are a helpful assistant that decides if a stats question needs clarification, "
            "document search, or can be answered directly.\n\n"
            "Return one of: 'simple', 'search', or 'clarify'.\n"
            "- Use 'clarify' if you think a follow-up question is needed to give a better answer."
        )
        response = llm_client.invoke(f"{system_prompt}\nQuestion: {question}")
        decision = response.strip().lower()
    else:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that decides if a stats question needs clarification, "
                        "document search, or can be answered directly.\n\n"
                        "Return one of: 'simple', 'search', or 'clarify'.\n"
                        "- Use 'clarify' if you think a follow-up question is needed to give a better answer."
                    )
                },
                {"role": "user", "content": f"Question: {question}"}
            ]
        )
        decision = response.choices[0].message.content.strip().lower()
    return {"intent": decision}


def ask_clarification(state):
    """Ask a follow-up question and wait for user's answer."""
    question = state["question"]

    # Ask a clarifying question based on input
    if PROVIDER == "Ollama":
        follow_up = llm_client.invoke(
            "Ask a follow-up question that helps clarify what statistical test to use.\n" + question
        ).strip()
    else:
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ask a follow-up question that helps clarify what statistical test to use."},
                {"role": "user", "content": question}
            ]
        )
        follow_up = response.choices[0].message.content.strip()
    print(f"\n🤖 Clarifying Question: {follow_up}")
    user_reply = input("💬 Your Answer: ")
    full_question = f"{question} User clarified: {user_reply}"
    return {"question": full_question}


def retrieve_info(state):
    """Use the RAG tool to answer from embedded docs."""
    question = state["question"]
    answer = get_doc_answer(question)
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

    if PROVIDER == "Ollama":
        response = llm_client.invoke(prompt)
        return {"code_snippet": response.strip()}
    else:
        response = llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"code_snippet": response.choices[0].message.content.strip()}