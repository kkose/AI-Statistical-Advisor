import os
import time
import streamlit as st

# Config page
st.set_page_config(page_title="Stats Advisor Agent",
                   page_icon='🤖',
                   layout="wide",
                   initial_sidebar_state="expanded")

# Add a place to enter the API key
with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY", type="password")

    # Save the API key to the environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Clear
    if st.button('Clear'):
        st.rerun()
    
    st.divider()
    # About
    st.write("Designed with :heart: by [Gustavo r Santos](https://gustavorsantos.me)")

# Title and Instructions
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    
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
