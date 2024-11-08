### Import dependencies

import os
from dotenv import load_dotenv

import pandas as pd
import tiktoken

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

## for timestamping save files
from datetime import datetime

## for asynch function calling
import asyncio

## for post-processing helper functions
import local_helper_functions as hf

## for browser app
import streamlit as st
import traceback
import time
from io import BytesIO
from docx import Document
import pickle

st.set_page_config(page_title="Local Search",
    page_icon="🔍"
    )

### Intro blurb

st.write("## 🔍 Local Search")

st.write(
    "👈 Get started by selecting how you would like the answers to your queries to be structured."
    )

### Set static variables (uncomment for private apps)

## Load .env file to environment and get keys
# load_dotenv()
# api_key = os.getenv('YOUR_API_KEY')

# ## Set model preferences
# model = 'gpt-4o-mini' # Or preferred OpenAI model
# api_type = 'OpenaiApiType.OpenAI' # Or OpenaiApiType.AzureOpenAI

# ## Set input directory path (replace with your GraphRAG pipeline output path)
# INPUT_DIR = './output/YYYYMMDD-HHMMSS/artifacts'

### Sidebar input variables
with st.sidebar:

    with st.form("configuration_form"):

        st.markdown("**Configure your search:**")
        api_key = st.text_input("Your API key", type="password")
        model = st.selectbox(
            "Select your query model",
            options=['gpt-4o-mini', 'gpt-4o', 'gpt-4o-2024-08-06', 'o1-preview', 'o1-mini']
            )
        api_type = st.radio(
            "Select your API type",
            options=['OpenaiApiType.OpenAI', 'OpenaiApiType.AzureOpenAI'],
            horizontal=True
            )
        INPUT_DIR = st.text_input(
            "Knowledge graph location", 
            help="Path to your local GraphRAG pipeline output (eg. _./output/YYYYMMDD-HHMMSS/artifacts_)."
            )
        response_type = st.text_input(
            "Response type", "multiple paragraphs",
            help="Can be anything at all, eg. matrix table, single paragraph, multiple paragraph, etc."
            )
        config_submitted = st.form_submit_button("Press to configure your search")
        if config_submitted:
            st.write("Search configured!")

if config_submitted:

    ### Load and cache community reports as context for local search

    @st.cache_data
    # define function to load data
    def load_data(INPUT_DIR):

        ### Load indexing reports for context
        INPUT_DIR = INPUT_DIR
        LANCEDB_URI = f"{INPUT_DIR}/lancedb"

        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"
        RELATIONSHIP_TABLE = "create_final_relationships"
        COVARIATE_TABLE = "create_final_covariates"
        TEXT_UNIT_TABLE = "create_final_text_units"
        FINAL_DOCS_TABLE = "create_final_documents" # for referencing only
        COMMUNITY_LEVEL = 2

        #### Read entities
        # read nodes table to get community and degree data
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        # load description embeddings to an in-memory lancedb vectorstore
        # to connect to a remote db, specify url and port values.
        _description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings",
        )
        _description_embedding_store.connect(db_uri=LANCEDB_URI)
        entity_description_embeddings = store_entity_semantic_embeddings(
            entities=entities, vectorstore=_description_embedding_store
        )

        st.sidebar.write(f"Entity count: {len(entity_df)}")

        #### Read relationships
        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        st.sidebar.write(f"Relationship count: {len(relationship_df)}")

        #### Read covariates: activate only after indexing with prompt tuning
        # See 'claim_extraction' in pipeline settings file
        covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")

        claims = read_indexer_covariates(covariate_df)

        st.sidebar.write(f"Claim records: {len(claims)}")
        covariates = {"claims": claims}

        #### Read community reports
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

        st.sidebar.write(f"Report records: {len(report_df)}")

        #### Read text units
        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        st.sidebar.write(f"Text unit records: {len(text_unit_df)}")

        #### Read documents (for referencing)
        documents_df = pd.read_parquet(f"{INPUT_DIR}/{FINAL_DOCS_TABLE}.parquet")

        return (
            entity_df, entities, _description_embedding_store, relationship_df, relationships,
            covariates, reports, text_unit_df, text_units, documents_df
            )

    # Execute function to load and cache data
    try:
        (
        entity_df, entities, _description_embedding_store, relationship_df, relationships,
        covariates, reports, text_unit_df, text_units, documents_df
        ) = load_data(INPUT_DIR)

    except Exception as e:
        st.error(f"""There was an error loading your data.

            Please check that the file path you entered is correct and uses forward
            slashes (.../...) !

            Error: {e}

            Traceback: {traceback.format_exc()}
            """)

    # Define function to load and cache search engine config parameters
    @st.cache_resource
    def configure_search_engine(
        api_key, model, api_type, reports, text_units, entities,
        relationships, covariates, response_type, _description_embedding_store
        ):

        ### Set API key

        api_key = api_key

        ### Initiate model

        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            api_type=api_type
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        text_embedder = OpenAIEmbedding(
            api_key=api_key,
            api_base=None,
            api_type=api_type,
            model="text-embedding-3-small",
            deployment_name="text-embedding-3-small",
            #max_retries=20,
        )

        #### Create local search context builder

        local_context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            # if did not run covariates during indexing, set this to None
            covariates=covariates, # None or covariates
            entity_text_embeddings=_description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )

        ### Configure parameters for global search

        #### Set parameters

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
            "max_tokens": 12_000,
        }

        llm_params = {
            "max_tokens": 2_000,
            "temperature": 0.0,
        }

        #### Configure search engine

        local_search_engine = LocalSearch(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type=response_type,  # free form text describing the response type and format, can be anything, e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
        )

        return local_search_engine

    # Execute function to load and cache the search engine
    try:
        local_search_engine = configure_search_engine(
        api_key, model, api_type, reports, text_units, entities,
        relationships, covariates, response_type, _description_embedding_store
            )

    except Exception as e:
        st.error(f"""There was an error configuring the search engine.

            Error: {e}

            Traceback: {traceback.format_exc()}
            """)

    ## Define initial session states

    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = ["..."]

    if 'query_results' not in st.session_state:
        st.session_state['query_results'] = []

    if 'default_responses' not in st.session_state:
        st.session_state['default_responses'] = ["..."]

    if 'sources_output' not in st.session_state:
        st.session_state['sources_output'] = ["..."]

    #### Query time

    @st.fragment
    def query_time():

        ## Define containers
        query_container = st.container()
        default_expander = st.expander("🔽 The answer to your query",
            #expanded=True
            )
        sources_expander = st.expander("🔽 Source texts")
        results_object_expander = st.expander("⚠️ Results object (for deeper analysis)")

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        async def perform_search_with_progress(query, progress_bar):

            # Start time measurement
            start_time = time.time()
            
            # Estimate total search time (set an estimated upper bound based on previous experience)
            estimated_total_time = 20  # adjust for typical query duration in seconds

            # Run the search function in an asyncio event loop, with regular progress updates
            search_task = asyncio.create_task(local_search_engine.asearch(query))
            
            # While the task is running, update the progress bar based on elapsed time
            while not search_task.done():
                elapsed_time = time.time() - start_time
                progress = min(int((elapsed_time / estimated_total_time) * 100), 100)
                progress_bar.progress(progress, "Working on your query!")
                await asyncio.sleep(0.1)  # adjust for the frequency of bar updates

            # Wait for the search to finish and get the result
            result = await search_task
            progress_bar.progress(100, "Nearly there!")  # Ensure progress is set to 100% on completion

            return result

        # User feedback widgets
        def processing_query():
            successful_submission = query_container.success(
                "Query submitted, working on it!", icon="✅")
            time.sleep(3) # Wait for 3 seconds
            successful_submission.empty() # Clear the success message

        # Download generated responses as .docx 
        def get_docx(text):
            document = Document()
            document.add_paragraph(text)
            byte_io = BytesIO()
            document.save(byte_io)
            return byte_io.getvalue()

        # Take query input
        with query_container:

            with st.form(
                "query_form", clear_on_submit=True):
                query = st.text_area("Type your query:", label_visibility="collapsed",
                placeholder="""Type your query here. Once a response has been generated, click on the
                tabs below to expand them and read the answers."""
                )
                query_submitted = st.form_submit_button("Submit query", on_click=processing_query)

        if query: # on query only so that subsequent expanders persist (button True state is ephemeral)

            if st.session_state['query_history'][-1] != query:

                st.session_state['query_history'].append(query)

                # Initialize progress bar
                progress_bar = query_container.progress(0, "Working on your query!")  # Start progress at 0%

                # Perform local search
                try:
                    # Execute search with progress updates
                    result = asyncio.run(perform_search_with_progress(query, progress_bar))

                except Exception as e:
                    st.error(f"""There was a problem executing the query.

                        Error: {e}

                        Traceback: {traceback.format_exc()}
                        """)

                # Create traceability tables of entity IDs, relationship IDs,
                # text units, and source titles
                try:

                    traceability_table, source_frequency_table = hf.create_traceability(
                        entity_df, relationship_df,
                        text_unit_df, documents_df, result
                        )

                    results_dict = {
                    query: [result, traceability_table, source_frequency_table]
                    }

                    st.session_state['query_results'].append(results_dict)

                except Exception as e:
                    st.error(f"""There was a problem processing the response.

                        Error: {e}

                        Traceback: {traceback.format_exc()}
                        """)

                # Remove progress bar after completion
                progress_bar.empty()

                # Rerun to update session state
                st.rerun(scope="fragment")

            # Format default response (+ most frequent sources)
            try:
                default_output = hf.combine_query_response_sources(
                    st.session_state['query_history'][-1],
                    st.session_state['query_results'][-1][st.session_state['query_history'][-1]][0],
                    st.session_state['query_results'][-1][st.session_state['query_history'][-1]][2]
                    )

                st.session_state['default_responses'].append(default_output)

            except Exception as e:
                st.error(f"""There was a problem formatting the response.

                    Error: {e}

                    Traceback: {traceback.format_exc()}
                    """)

            with default_expander:

                st.download_button(
                    "Download the response.",
                    data=get_docx(default_output),
                    file_name=f"Local_search-Response-{current_time}.docx",
                    mime="docx"
                    )

            # Format text sources for returned output
            with sources_expander:

                try:
                    text_unit_output = hf.retrieve_text_units(
                        st.session_state['query_history'][-1],
                        st.session_state['query_results'][-1][st.session_state['query_history'][-1]][1]
                        )

                    st.session_state['sources_output'].append(text_unit_output)

                except Exception as e:
                    st.error(f"""There was a problem formatting the sources.

                        Error: {e}

                        Traceback: {traceback.format_exc()}
                        """)

                with sources_expander:

                    st.download_button("Download the sources.",
                        text_unit_output,
                        file_name=f"Local_search-Sources-{current_time}.txt")

            # Optionally download result object for analysis
            with results_object_expander:

                st.download_button("Download the results object.",
                    data=pickle.dumps(st.session_state['query_results'][-1][st.session_state['query_history'][-1]][0]),
                    file_name=f"Local_search-Result_object-{current_time}.pickle")

        # Load query page with persistent expanders for responses
        default_expander.markdown(st.session_state['default_responses'][-1])
        sources_expander.markdown(st.session_state['sources_output'][-1])

    query_time()