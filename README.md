# Easy GraphRAG Queries 💡

This app came about to make a research project a little tidier and for people more comfortable with their browser than a Python IDE 🐍.

It wraps the Global and Local search methods of [GraphRAG v0.3.6](https://pypi.org/project/graphrag/0.3.6/) in a Streamlit UI. It was made for exploring a [GraphRAG knowledge graph](https://microsoft.github.io/graphrag/) in a local (private) directory, from the comfort of a browser. It works with the [OpenAI API](https://platform.openai.com/docs/overview) 🤖.

You can select and configure the Global and Local search engines, submit queries, and view  the results in your browser. You can also view and download supporting AI-generated reports, source document references, as well as the results object.

### Why bother?
- **User-friendliness**: Streamline querying and saving the outputs of research projects from a browser.
- **Referencing**: Approximate conventional referencing for academic or professional research projects.

### Key features:
- **Search configuration**: Select a search method, the preferred OpenAI model, and how the response should be presented.
- **References**: The generated response references the most-used source documents for exploration and corroboration.
- **Supporting analyses**: Get AI-generated reports to support the response.
- **Download results**: Download query results, analyses, and sources.

### How to use it:
1. **Clone the repo**: Clone this repo to your local GraphRAG project directory.
2. **GraphRAG pipeline**: If you haven't already, follow the [GraphRAG docs](https://microsoft.github.io/graphrag/get_started/) to initiate your project environment and create your knowledge graph.
3. **Install requirements**: Install any requirements you might need in a virtual environment.
4. **Run the app**: Run the app in your GraphRAG project directory:

```shell
streamlit run Graph_query.py
```

5. **Configure search**: Enter your API key, choose a model, and specify the data source (a GraphRAG pipeline output folder).
6. **Submit query**: Type your query and submit it.
7. **View results**: Click on the expanders to see the response, supporting analyses, and sources.
8. **Download**: Download the results for your research records or to explore later.

### Requirements
Install the required dependencies in a virtual environment if needed:

```shell
pip install -r requirements.txt
```

### Acknowledgements
This app was inspired by the [Microsoft GraphRAG project](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/).

### License
This app is open source and published under the MIT License. There will be no maintenance but look forward to people improving it!

Happy searching! ✨