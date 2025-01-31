
import numpy as np  
import pandas as pd 

from .col_mappings import * 

def post_process(df):
    
    cols = df.columns.tolist()
    perc_cols = [x for x in cols if 'perc_' in x ]

    # fillna for perc cols with 0 
    df[perc_cols] = df[perc_cols].fillna(0)

    # if mean energy consumption is nan then make sum nan 
    df.loc[df['mean_ENERGY_CONSUMPTION_CURRENT'].isnull(), 'sum_ENERGY_CONSUMPTION_CURRENT'] = None 
    df.loc[df['mean_ENERGY_CONSUMPTION_CURRENT'].isnull(), 'sum_energy_consumption_total'] = None 
    df = aggregate_postcode_data(df)
    
    age_mapping, tenure_mapping, bf_mapping, transaction_mapping = get_mappings(df)    
    print('Mappings:', age_mapping, tenure_mapping, bf_mapping, transaction_mapping)    

    result_df = process_all_mappings(
        df,
        age_mapping,
        tenure_mapping,
        bf_mapping,
        transaction_mapping
    )
    print(result_df.columns.tolist() )

    return pd.concat([df , result_df], axis=1 ), age_mapping


def aggregate_postcode_data(df):
    """
    Aggregates postcode data based on column prefixes with correct weighted calculations.
    """
    # Get postcodes with multiple entries
    postcode_counts = df['postcode'].value_counts().reset_index()
    postcode_counts.columns = ['postcode', 'count']
    multiple_postcodes = postcode_counts[postcode_counts['count'] > 1]['postcode'].tolist()
    
    # Split dataframe into single and multiple entries
    df_multiple = df[df['postcode'].isin(multiple_postcodes)].copy()
    df_single = df[~df['postcode'].isin(multiple_postcodes)].copy()
    
    # If no multiple entries, return original df
    if len(df_multiple) == 0:
        return df
    
    # Group columns by type
    sum_cols = [col for col in df.columns if col.startswith('sum_')] + ['count_uprn', 'count_building_ref']
    mean_cols = [col for col in df.columns if col.startswith('mean_')]
    perc_cols = [col for col in df.columns if col.startswith('perc_')]
    median_cols = [col for col in df.columns if col.startswith('median_')]
    null_cols = [col for col in df.columns if col.startswith('null_')]
    
    # Create aggregation dictionary
    agg_dict = {}
    
    # Sum columns - simple sum
    for col in sum_cols + null_cols:
        agg_dict[col] = 'sum'
    
    # Count buildings - needed for weighted calculations
    agg_dict['count_buildings'] = 'sum'
    
    # Perform initial aggregation
    df_agg = df_multiple.groupby('postcode').agg(agg_dict).reset_index()
    
    # Handle weighted means - FIXED
    for col in mean_cols:
        df_agg[col] = df_multiple.groupby('postcode').apply(
            lambda x: np.average(x[col], weights=x['count_buildings'])
        ).reset_index(name=col)[col]
    
    # Handle weighted percentages - FIXED
    for col in perc_cols:
        df_agg[col] = df_multiple.groupby('postcode').apply(
            lambda x: np.average(x[col], weights=x['count_buildings'])
        ).reset_index(name=col)[col]
    
    # Set median columns to null
    for col in median_cols:
        df_agg[col] = None
    
    # Combine with single entry postcodes
    result_df = pd.concat([df_single, df_agg], ignore_index=True)
    
    return result_df

# apply mappings 


def reverse_and_combine_mappings(*mappings):
    """
    Reverses and combines mappings to group old columns by new column name.
    
    Args:
        *mappings: Variable number of mapping dictionaries where
                  key = old column name, value = new column name
    
    Returns:
        dict: Combined mapping where key = new column name, value = list of old column names
    """
    reversed_mapping = {}
    
    for mapping in mappings:
        for old_col, new_col in mapping.items():
            if new_col not in reversed_mapping:
                reversed_mapping[new_col] = []
            reversed_mapping[new_col].append(old_col)
    
    return reversed_mapping

def apply_mapping(df, mapping):
    """
    Applies the mapping to create new columns by summing old columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        mapping (dict): Dictionary where keys are old column names 
                       and values are new column names
    
    Returns:
        pd.DataFrame: New dataframe with summed columns
    """
    # First reverse the mapping to group by new column names
    reversed_mapping = reverse_and_combine_mappings(mapping)
    
    # Create new dataframe
    result = pd.DataFrame(index=df.index)
    
    # For each new column, sum all corresponding old columns
    for new_col, old_cols in reversed_mapping.items():
        # Filter for columns that exist in the dataframe
        valid_cols = [col for col in old_cols if col in df.columns]
        if valid_cols:
            result[new_col] = df[valid_cols].sum(axis=1)
    
    return result

 
def process_all_mappings(df, age_mapping, tenure_mapping, bf_mapping, transaction_mapping):
    """
    Process all mappings in sequence.
    
    Args:
        df (pd.DataFrame): Input dataframe
        *mappings: Various mapping dictionaries
    
    Returns:
        pd.DataFrame: Processed dataframe with all mappings applied
    """
    # Combine all mappings into one
    all_mappings = {}
    for mapping in [age_mapping, tenure_mapping, bf_mapping, transaction_mapping]:
        all_mappings.update(mapping)
    
    # Apply the combined mapping
    return apply_mapping(df, all_mappings)

def get_mappings(df):
    cols = df.columns.tolist()  

    age = [x for x in cols if 'CONSTRUCTION_AGE_BAND' in x] 
    tenure = [x for x in cols if 'TENURE' in x ] 
    built_form = [x for x in cols if 'BUILT_FORM' in x]    
    transaction_type = [x for x in cols if 'TRANSACTION_TYPE' in x]

    age_mapping = get_mapping_for_age_columns(age)
    tenure_mapping = get_tenure_mapping_for_columns(tenure)
    bf_mapping = get_built_form_mapping_for_columns(built_form)
    transaction_mapping = get_transaction_type_mapping_for_columns(transaction_type)    
    return age_mapping, tenure_mapping, bf_mapping, transaction_mapping