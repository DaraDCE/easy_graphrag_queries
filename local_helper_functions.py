## Import dependencies for local search post-processing
import pandas as pd
import re

### 1. Define helper functions for post-processing of response object:

#### 1.1.  Functions to map generated response to source documents

def map_reports_entities_relationships(response_str):
    """Helper function to extract entity and relationship IDs and map them,
    ensuring that no surplus values are dropped."""
    
    # Declare lists to store mappings
    entity_id_list = []
    relationship_id_list = []
    
    # Process the response object string
    if response_str:
        # Extract entities and relationships from the content string
        entity_groups = re.findall(r'Entities \(([^)]+)\)', response_str)
        relationship_groups = re.findall(r'Relationships \(([^)]+)\)', response_str)
        
        # Process entities
        entity_ids = []
        for entity_ids_str in entity_groups:
            entity_ids += [id.strip() for id in entity_ids_str.split(', ') if id.isdigit()]
        
        # Process relationships
        relationship_ids = []
        for relationship_ids_str in relationship_groups:
            relationship_ids += [id.strip() for id in relationship_ids_str.split(', ') if id.isdigit()]

        entity_count = len(entity_ids)
        relationship_count = len(relationship_ids)

        # Iterate through the maximum of entity or relationship counts to ensure all values are processed
        for i in range(max(entity_count, relationship_count)):
            # Get entity ID if it exists, otherwise assign 'incomplete match'
            entity_id = entity_ids[i] if i < entity_count else 'incomplete match'
            
            # Get relationship ID if it exists, otherwise assign 'incomplete match'
            relationship_id = relationship_ids[i] if i < relationship_count else 'incomplete match'
            
            # Append mappings
            entity_id_list.append(entity_id)
            relationship_id_list.append(relationship_id)
    
    # Create a dataframe that maps 'entity_id' and 'relationship_id'
    traceability_df = pd.DataFrame({
        'entity_id': entity_id_list,
        'relationship_id': relationship_id_list
    })
    
    return traceability_df

def map_text_unit_ids(traceability_df, df_relationships, df_entities):
    """Helper function to map 'text_unit_ids' and 'source_id' into the same 'text_unit_ids' column
    and onto traceability table"""

    # Map 'text_unit_ids' from df_relationships to traceability table using 'relationship_id' and 'human_readable_id'

    # Flatten numpy arrays in df_relationships['text_unit_ids'] into individual rows with string values
    df_relationships_exploded = df_relationships[['human_readable_id', 'text_unit_ids']].explode('text_unit_ids')
    df_relationships_exploded['text_unit_ids'] = df_relationships_exploded['text_unit_ids'].astype(str)

    # Merge the flattened relationship text_unit_ids into traceability_df on 'relationship_id' and 'human_readable_id'
    traceability_df = traceability_df.merge(df_relationships_exploded, 
                                            left_on=['relationship_id'], 
                                            right_on=['human_readable_id'], 
                                            how='left')

    # Map 'source_id' from df_entities to the same 'text_unit_ids' column using 'entity_id' and 'human_readable_id'

    # Flatten df_entities['source_id'] (comma-separated) into individual rows with string values
    df_entities['source_id'] = df_entities['source_id'].astype(str)
    df_entities_exploded = df_entities.assign(source_id=df_entities['source_id'].str.split(',')).explode('source_id')
    df_entities_exploded['source_id'] = df_entities_exploded['source_id'].astype(str).str.strip()

    # Merge the flattened entity source_ids into traceability_df on 'entity_id' and 'human_readable_id'
    traceability_df = traceability_df.merge(df_entities_exploded[['human_readable_id', 'source_id']],
                                            left_on=['entity_id'], 
                                            right_on=['human_readable_id'], 
                                            how='left')

    # Combine 'text_unit_ids' from relationships and 'source_id' from entities into a single column
    # Fill missing values for 'text_unit_ids' and 'source_id' as 'no direct match'
    traceability_df['text_unit_ids'] = traceability_df.apply(
        lambda row: row['text_unit_ids'] if pd.notna(row['text_unit_ids']) else row['source_id'],
        axis=1
    ).fillna('no direct match')

    # Clean up redundant columns
    traceability_df = traceability_df.drop(columns=['human_readable_id_x', 'human_readable_id_y', 'source_id'], errors='ignore')
    
    # Drop duplicate rows
    traceability_df = traceability_df.drop_duplicates(ignore_index=True)

    # Return the traceability dataframe along
    return traceability_df

def map_text_unit_content(df_final_text_units, traceability_df): 
    """Helper function to map 'text' column of df_final_text_units onto the traceability table"""
    
    # Merge df_final_text_units with traceability_df on 'text_unit_ids' (traceability_df) and 'id' (df_final_text_units)
    traceability_df = traceability_df.merge(df_final_text_units[['id', 'text']],
                                            left_on='text_unit_ids',
                                            right_on='id',
                                            how='left')

    # Drop the redundant 'id' column after merging
    traceability_df.drop('id', axis=1, inplace=True)
    
    # Fill in blank or missing 'text' values with 'incomplete match'
    traceability_df['text'] = traceability_df['text'].fillna('incomplete match')
    
    # Rename column
    traceability_df.rename(columns = {'text':'text_unit_content'}, inplace = True)

    # Return the updated traceability dataframe
    return traceability_df

def map_titles(df_documents, traceability_df): 
    """ Helper function to map 'title' column of df_documents onto the traceability table"""
    
    # Flatten 'text_units' in df_documents to match with 'text_unit_ids' in traceability_df
    df_documents_exploded = df_documents.explode('text_unit_ids').reset_index(drop=True)

    # Merge df_documents_exploded with traceability_df on 'text_unit_ids' (traceability_df) and 'text_units' (df_documents)
    traceability_df = traceability_df.merge(df_documents_exploded[['text_unit_ids', 'title']],
                                            left_on='text_unit_ids',
                                            right_on='text_unit_ids',
                                            how='left')
    
    # Fill in any blank or missing 'title' values with 'incomplete match'
    traceability_df['title'] = traceability_df['title'].fillna('incomplete match')

    # Return the updated traceability dataframe
    return traceability_df

def most_frequent_sources(traceability_table):
    """Helper function to find most frequent source documents matching report ids in generated response""" 
    
    # Count the frequency of each value in 'title' in the traceability_df
    frequency_df = traceability_table['title'].value_counts().reset_index(name='title_frequency')
    frequency_df.columns = ['title', 'title_frequency']
    
    # Filter top quartile by frequency (>= 50th percentile)
    above_median_df = frequency_df[frequency_df['title_frequency'] >= frequency_df[
    'title_frequency'].quantile(0.5)]
    
    # return the traceability table reduced by the top quartile of matched source documents
    return above_median_df

#### 1.2. Main function to produce traceability table and other reference tables

def create_traceability(entity_df, relationship_df, text_unit_df, documents_df, result):
    """Primary function to map generated responses to source documents and return traceability tables"""
    
    # Stage dataframes
    df_entities = entity_df
    df_relationships = relationship_df
    df_final_text_units = text_unit_df
    df_documents = documents_df
    
    # Declares lists to store final results
    all_results = []
    reduced_results = []
    
    # Convert 'human_readable_id' series from int to str for downstream processing
    df_entities['human_readable_id'] = df_entities['human_readable_id'].astype(str)
    
    if result:
        response_str = result.response

        # 1. Initiate traceability table with mapping of relationship_id -> entity_id -> report_id
        initial_traceability_df = map_reports_entities_relationships(response_str)
        
        # 2. Update traceability with mapping of relationship_id, entity_id -> text_unit_ids, source_id
        first_intermediate_traceability_df = map_text_unit_ids(initial_traceability_df, df_relationships, df_entities)
        
        # 3. Update traceability with mapping of text_unit_id -> text
        second_intermediate_traceability_df = map_text_unit_content(df_final_text_units, first_intermediate_traceability_df)

        # 4: Map 'title' column of df_documents onto the traceability table
        final_traceability_df = map_titles(df_documents, second_intermediate_traceability_df)
        
        # 5: Create table of most frequent source
        frequency_df = most_frequent_sources(final_traceability_df)

    # Return traceability table and other useful tables
    return final_traceability_df, frequency_df

#### 1.3. Function to return default formatted response with sources - DEFAULT

def combine_query_response_sources(query, result_object, frequency_df):
    """Function to combine and format query + generated response + most frequent sources"""
    
    # Declare variable to hold formatted string outputs
    references = ""
    
    # Format string of source document references from frequency table
    for i, source in enumerate(frequency_df['title']):
        references += f"{i+1}. {source}\n"
        
    # Format string of query, elements of result object, and most frequent sources
    query_response_sources = (
        f"**Query:** {query}\n\n"
        f"**Generated response:**\n\n{result_object.response}\n\n"
        f"**Most-referenced source documents:**\n\n{references}\n"
        f"**Performance**\n\nLLM calls: {result_object.llm_calls}\n"
        f"\nLLM tokens: {result_object.prompt_tokens}\n"
        f"\nCompletion time: {round(result_object.completion_time, 2)}\n"
    )

    return query_response_sources

#### 1.4. Function to return source text units - OPTIONAL

def retrieve_text_units(query, traceability_table):
    """Function to retrieve text units corresponding to response"""
    
    # Declare variable to hold formatted string outputs
    text_units = ""

    # Drop duplicate rows
    traceability_table_drop = traceability_table.drop_duplicates(ignore_index=True)
    
    # Filter to series of interest
    filtered_table = traceability_table_drop[['text_unit_content', 'title']]

    # Append foramtted query to string
    text_units += f"\n**Query:** {query}\n\nThe query was based on the following texts:\n"
                
    # Append text units related to the current query
    for _, row in filtered_table.iterrows():
        text_units += f"\n**Source:** {row['title']}\n\n**Source text snippet:**\n\n{row['text_unit_content']}\n\n---\n\n"
 
    # Signal end of report        
    text_units += f"\n### END OF REPORT ###"
    
    # Return the matched text units
    matched_text_units = (
        f"**The following are the original text snippets on which the generated response was based:**\n"
        f"{text_units}"
    )

    return matched_text_units