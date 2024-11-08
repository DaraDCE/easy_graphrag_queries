import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="💡",
)

st.write("# Query your Knowledge Graph! 💡")
st.sidebar.success("Select Global or Local search.")

st.markdown(
    """
    Use this app to conveniently query your locally stored [Microsoft GraphRAG](https://microsoft.github.io/graphrag/).
    Find out more about building a knowledge graph with **Microsoft GraphRAG** on your own computer
    [here](https://microsoft.github.io/graphrag/posts/get_started/).

    ⚠️ This app was implemented with [GraphRAG v0.3.6](https://pypi.org/project/graphrag/0.3.6/)!

    👈 Get started by selecting either **Global** or **Local** search on the left.

    ### What are **Global** and **Local** search?

    - With Global search, the AI model essentially attempts to answer your question by
    looking at the **entire** data represented in the knowledge graph. It's a more intensive
    (and expensive) operation but is especially good for getting thematic or holistic overviews.
    Find out more about it [here](https://microsoft.github.io/graphrag/posts/query/notebooks/global_search_nb/).
    
    - With Local search, the AI model combines relevant data represented in the knowledge graph
    with snippets of source data to answer your question. It is less intensive and well-suited
    for answering specific questions about the data. Find out more [here](
    https://microsoft.github.io/graphrag/posts/query/notebooks/local_search_nb/).

    👈 Select **Global** or **Local** search to get started!

"""
)