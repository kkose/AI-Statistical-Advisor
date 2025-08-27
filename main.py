import os
import time
import streamlit as st
from typing import Optional

# Config page
st.set_page_config(page_title="Stats Advisor Agent",
                   page_icon='🤖',
                   layout="wide",
                   initial_sidebar_state="expanded")

# Add a place to enter the API key
with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY", type="password")

with st.sidebar:
    provider = st.selectbox("Select LLM Provider", ["OpenAI", "Ollama"], index=0)
    api_key: Optional[str] = None
    ollama_model: Optional[str] = None
    # If OpenAI selected, ask for key
    if provider == "OpenAI":
        api_key = st.text_input("OPENAI_API_KEY", type="password", key="openai_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            # If no key, fallback to Ollama
            provider = "Ollama"
            st.info("No OpenAI API key provided. Defaulting to Ollama.")
    if provider == "Ollama":
        ollama_model = st.text_input("Ollama Model Name", value="llama3", key="ollama_model")
        if ollama_model:
            os.environ["OLLAMA_MODEL"] = ollama_model

    if st.button('Clear'):
        st.rerun()
    st.divider()
    st.write("Designed with :heart: by [Gustavo R. Santos](https://gustavorsantos.me)")

# Title and Instructions
if provider == "Ollama" and not ollama_model:
    st.warning("Please enter your Ollama model name in the sidebar.")
st.title('Statistical Advisor Agent | 🤖')
st.caption('This AI Agent is trained to answer questions about statistical tests from the [Scipy](https://docs.scipy.org/doc/scipy/reference/stats.html) package.')
st.caption('Ask questions like: "What is the best statistical test to compare two means".')
st.divider()


# User question
question = st.text_input(label="Ask me something:",
                         placeholder= "e.g. What is the best test to compare 3 groups means?")


# Run the graph
if st.button('Search'):
    
    # Progress bar
    progress_bar = st.progress(0)

    with st.spinner("Thinking..", show_time=True):
        
        from langgraph_agent.graph import build_graph
        progress_bar.progress(10)
        # Build the graph
        graph = build_graph()
        result = graph.invoke({"question": question})
        
        # Progress bar
        progress_bar.progress(50)

        # Print the result
        st.subheader("📖 Answer:")
        
        # Progress bar
        progress_bar.progress(100)

        st.write(result["final_answer"])

        if "code_snippet" in result:
            st.subheader("💻 Suggested Python Code:")
            st.write(result["code_snippet"])
