import pandas as pd 



def generate_mean_attribute_for_numeric_col(pc, data, col):
    # Calculate mean for the specified column
    try:
        # Handle potential NaN/null values by dropping them
        mean_value = data[col].dropna().mean()
        
        # Create dictionary with results
        results = {
            f"mean_{col}": round(mean_value, 2),  # Round to 2 decimal places
            'postcode': pc
        }
        
        return results
        
    except (TypeError, ValueError) as e:
        # Handle cases where column might not be numeric
        print(f"Error calculating mean for column {col}: {str(e)}")
        return {
            f"mean_{col}": None,
            'postcode': pc
        }
    

def generate_percentage_atr(pc, data, col):
    # Calculate total number of buildings
    count_builds = data.shape[0]
    
    # Calculate percentages
    results = (data.groupby([col])['LMK_KEY'].count() / count_builds * 100).to_dict()
    
    # Create new dictionary with modified keys
    updated_results = {f"{col}_{key}": value for key, value in results.items()}
    
    # Add postcode
    updated_results['postcode'] = pc
    
    return updated_results


def process_postcode_attributes(data, postcode, cols):
    # Filter dataframe for specific postcode
    postcode_data = data[data['POSTCODE'] == postcode].copy()
    
    if postcode_data.empty:
        print(f"No data found for postcode: {postcode}")
        return None
    
    # Define numeric and categorical columns
    numeric_cols = [
        'CURRENT_ENERGY_EFFICIENCY',
        'POTENTIAL_ENERGY_EFFICIENCY',
        'ENVIRONMENT_IMPACT_CURRENT',
        'ENVIRONMENT_IMPACT_POTENTIAL',
        'ENERGY_CONSUMPTION_CURRENT',
        'ENERGY_CONSUMPTION_POTENTIAL',
        'CO2_EMISSIONS_CURRENT',
        'CO2_EMISS_CURR_PER_FLOOR_AREA',
        'CO2_EMISSIONS_POTENTIAL',
        'NUMBER_HABITABLE_ROOMS'
    ]
    
    categorical_cols = [
        'CURRENT_ENERGY_RATING',
        'POTENTIAL_ENERGY_RATING',
        'UPRN_SOURCE',
        'TENURE',
        'BUILT_FORM'
    ]
    
    # Initialize results dictionary
    all_attributes = {}

    all_attributes['count_buildings'] = len(postcode_data)
    # Process numeric columns
    for col in numeric_cols:
        if col in cols:  # Only process if column is in requested cols
            mean_results = generate_mean_attribute_for_numeric_col(postcode, postcode_data, col)
            all_attributes.update(mean_results)
    
    # Process categorical columns
    for col in categorical_cols:
        if col in cols:  # Only process if column is in requested cols
            percentage_results = generate_percentage_atr(postcode, postcode_data, col)
            all_attributes.update(percentage_results)
    
    return all_attributes

# Example usage:
def process_multiple_postcodes(data, postcodes, cols):
    all_results = []
    for pc in postcodes:
        results = process_postcode_attributes(data, pc, cols)
        if results:
            all_results.append(results)
    
    return all_results


def analyse_epc(df, lad_code ):
    """ Takes an input df and calc all variables for all postcodes 
    """
    cols = [  'CURRENT_ENERGY_RATING',
    'POTENTIAL_ENERGY_RATING',
    'CURRENT_ENERGY_EFFICIENCY',
    'POTENTIAL_ENERGY_EFFICIENCY',
    'ENVIRONMENT_IMPACT_CURRENT',
    'ENVIRONMENT_IMPACT_POTENTIAL',
    'ENERGY_CONSUMPTION_CURRENT',
    'ENERGY_CONSUMPTION_POTENTIAL',
    'CO2_EMISSIONS_CURRENT',
    'CO2_EMISS_CURR_PER_FLOOR_AREA',
    'CO2_EMISSIONS_POTENTIAL',
    'UPRN_SOURCE', 
    'NUMBER_HABITABLE_ROOMS',
    'TENURE' 
    'BUILT_FORM', 
    ] 
    postcodes = df.POSTCODE.unique().tolist()
    results = process_multiple_postcodes(df, postcodes, cols)
    res_df = pd.DataFrame(results) 
    res_df['lad_code'] = lad_code   
    return res_df   

