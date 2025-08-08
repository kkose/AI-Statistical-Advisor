from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from langgraph_agent.nodes import classify_intent, retrieve_info, respond, generate_code


# Define the state schema (just a dictionary for now)
class TypedDictState(TypedDict):
    question: str
    intent: str
    rag_answer: str
    code_snippet: str
    final_answer: str

def build_graph():
    # Build the LangGraph flow
    builder = StateGraph(TypedDictState)

    # Add nodes
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_info", retrieve_info)
    builder.add_node("respond", respond)
    builder.add_node("generate_code", generate_code)

    # Define flow
    builder.set_entry_point("classify_intent")

    builder.add_conditional_edges(
        "classify_intent",
        lambda state: state["intent"],
        {
            "search": "retrieve_info",
            "simple": "respond"
        }
    )

    builder.add_edge("retrieve_info", "respond")
    builder.add_edge("respond", "generate_code")
    builder.add_edge("generate_code", END)

    return builder.compile()