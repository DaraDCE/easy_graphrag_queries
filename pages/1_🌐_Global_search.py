### Import dependencies

import os
from dotenv import load_dotenv

import pandas as pd
import tiktoken

from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.structured_search.global_search.search import GlobalSearch

## for timestamping save files
from datetime import datetime

## for asynch function calling
import asyncio

## for post-processing helper functions
import global_helper_functions as hf

## for browser app
import streamlit as st
import traceback
import time
from io import BytesIO
from docx import Document
import pickle

st.set_page_config(page_title="Global Search",
    page_icon="🌐"
    )

### Intro blurb

st.write("## 🌐 Global Search")

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

    ### Load and cache GraphRAG pipeline outputs as context for global search

    @st.cache_data
    # Define function to load data
    def load_data(INPUT_DIR):
        # parquet files generated from indexing pipeline (amend if different)
        INPUT_DIR = INPUT_DIR
        COMMUNITY_REPORT_TABLE = "create_final_community_reports"
        ENTITY_TABLE = "create_final_nodes"
        ENTITY_EMBEDDING_TABLE = "create_final_entities"

        # community level in the Leiden community hierarchy from which we will load the community reports
        # higher value means we use reports from more fine-grained communities (at the cost of higher computation cost)
        COMMUNITY_LEVEL = 2

        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        # additional files for referencing
        FINAL_DOCS_TABLE = "create_final_documents"
        RELATIONSHIPS_TABLE = "create_final_relationships"
        FINAL_TEXT_UNITS_TABLE = "create_final_text_units"

        df_documents = pd.read_parquet(f"{INPUT_DIR}/{FINAL_DOCS_TABLE}.parquet")
        df_relationships = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIPS_TABLE}.parquet")
        df_final_text_units = pd.read_parquet(f"{INPUT_DIR}/{FINAL_TEXT_UNITS_TABLE}.parquet")

        # st.sidebar.write(f"Total report count: {len(report_df)}")
        # st.sidebar.write(
        #     f"Report count after filtering by community level {COMMUNITY_LEVEL}: {len(reports)}"
        # )

        return (
            entity_df, report_df, entity_embedding_df, reports, entities,
            df_documents, df_relationships, df_final_text_units
            )

    # Execute function to load and cache data
    try:
        (entity_df, report_df, entity_embedding_df, reports, entities,
            df_documents, df_relationships, df_final_text_units) = load_data(INPUT_DIR)

    except Exception as e:
        st.error(f"""There was an error loading your data.

            Please check that the file path you entered is correct and uses forward
            slashes (.../...) !

            Error: {e}

            Traceback: {traceback.format_exc()}
            """)

    # Define function to load and cache search engine config parameters
    @st.cache_resource
    def configure_search_engine(api_key, model, api_type, reports, entities, response_type):

        ### Set API key

        api_key = api_key

        ### Initiate model

        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            api_type=api_type
        )

        token_encoder = tiktoken.get_encoding("cl100k_base")

        #### Build global context based on community reports

        context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,  # default to None if you don't want to use community weights for ranking
            token_encoder=token_encoder,
        )

        ### Configure parameters for global search

        #### Set parameters

        context_builder_params = {
            "use_community_summary": False,  # False means using full community reports. True means using community short summaries.
            "shuffle_data": True,
            "include_community_rank": True,
            "min_community_rank": 0,
            "community_rank_name": "rank",
            "include_community_weight": True,
            "community_weight_name": "occurrence weight",
            "normalize_community_weight": True,
            "max_tokens": 12_000,  # change this based on the token limit of the model (for a model with 8k limit, a good setting could be 5000)
            "context_name": "Reports",
        }

        map_llm_params = {
            "max_tokens": 1000,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        reduce_llm_params = {
            "max_tokens": 2000,  # change this based on the token limit of the model (for a model with 8k limit, a good setting could be 1000-1500)
            "temperature": 0.0,
        }

        #### Configure search engine

        search_engine = GlobalSearch(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            max_data_tokens=12_000,  # change this based on the token limit of the model (for a model with 8k limit, a good setting could be 5000)
            map_llm_params=map_llm_params,
            reduce_llm_params=reduce_llm_params,
            allow_general_knowledge=False,  # setting True will add instruction to encourage the LLM to incorporate general knowledge in the response, which may increase hallucinations, but could be useful in some use cases.
            json_mode=True,  # set this to False if LLM model does not support JSON mode.
            context_builder_params=context_builder_params,
            concurrent_coroutines=32,
            response_type=response_type,
        )

        return search_engine

    # Execute function to load and cache the search engine
    try:
        search_engine = configure_search_engine(api_key, model, api_type,reports, entities, response_type)

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

    if 'analysis_output' not in st.session_state:
        st.session_state['analysis_output'] = ["..."]

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
        analysis_expander = st.expander("🔽 Supporting AI analyses")
        sources_expander = st.expander("🔽 Source texts")
        results_object_expander = st.expander("⚠️ Results object (for further analysis)")

        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        async def perform_search_with_progress(query, progress_bar):

            # Start time measurement
            start_time = time.time()
            
            # Estimate total search time (set an estimated upper bound based on previous experience)
            estimated_total_time = 20  # adjust for typical query duration in seconds

            # Run the search function in an asyncio event loop, with regular progress updates
            search_task = asyncio.create_task(search_engine.asearch(query))
            
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

        def processing_reportids():
            successful_submission = sources_expander.success(
                "Report IDs submitted, working on it!", icon="✅")
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

                # Perform global search
                try:
                    # Execute search with progress updates
                    result = asyncio.run(perform_search_with_progress(query, progress_bar))

                except Exception as e:
                    st.error(f"""There was a problem executing the query.

                        Error: {e}

                        Traceback: {traceback.format_exc()}
                        """)

                # Create traceability tables of reports IDs, entity IDs, relationship IDs,
                # text units, and source titles
                try:

                    (traceability_table, relevant_context_df,
                    source_frequency_table) = hf.create_traceability(entity_df, df_relationships,
                    df_final_text_units, df_documents, result)

                    results_dict = {
                    query: [result, traceability_table, relevant_context_df, source_frequency_table]
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
                    st.session_state['query_results'][-1][st.session_state['query_history'][-1]][3]
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
                    file_name=f"Global_search-Response-{current_time}.docx",
                    mime="docx"
                    )

            # Format key supporting analyses
            try:
                analysis_output = hf.combine_query_supporting_analysis(
                    st.session_state['query_history'][-1],
                    st.session_state['query_results'][-1][st.session_state['query_history'][-1]][0],
                    st.session_state['query_results'][-1][st.session_state['query_history'][-1]][2]
                    )

                st.session_state['analysis_output'].append(analysis_output)

            except Exception as e:
                st.error(f"""There was a problem formatting the supporting analyses.

                    Error: {e}

                    Traceback: {traceback.format_exc()}
                    """)

            with analysis_expander:

                st.download_button("Download the analysis.",
                    analysis_output,
                    file_name=f"Global_search-Analysis-{current_time}.txt")

            # Optionally retrieve and format text sources for given report ID
            with sources_expander:

                with st.form("report_id_form", clear_on_submit=True):
                    # Format sources
                    report_ID_input = st.text_input("""Enter report IDs
                        to retrieve corresponding sources""",
                        placeholder="Eg. 123 OR 123, 456",
                        help="""Enter report IDs as digits
                        with commas separating them, eg. 1, 2, 3.
                        """)
                    ids_submitted = st.form_submit_button("Submit", on_click=processing_reportids)

                    if ids_submitted:

                        try:
                            text_unit_output = hf.retrieve_text_units_for_reportID(
                                report_ID_input,
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
                                file_name=f"Global_search-Sources-{current_time}.txt")

            # Optionally download result object for analysis
            with results_object_expander:

                st.download_button("Download the results object.",
                    data=pickle.dumps(st.session_state['query_results'][-1][st.session_state['query_history'][-1]][0]),
                    file_name=f"Global_search-Result_object-{current_time}.pickle")

        # Load query page with persistent expanders for responses
        default_expander.markdown(st.session_state['default_responses'][-1])
        analysis_expander.markdown(st.session_state['analysis_output'][-1])
        sources_expander.markdown(st.session_state['sources_output'][-1])

    query_time()