import pandas as pd 


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
    'NUMBER_HABITABLE_ROOMS',
        'NUMBER_HABITABLE_ROOMS',
'NUMBER_HEATED_ROOMS',
'TOTAL_FLOOR_AREA',
'EXTENSION_COUNT',
'energy_consumption_total',
'potential_energy_consumption_total',
] 

sum_cols = ['ENERGY_CONSUMPTION_CURRENT',
            'TOTAL_FLOOR_AREA',
            'energy_consumption_total',
            'potential_energy_consumption_total', ] 

categorical_cols = [
    'CURRENT_ENERGY_RATING',
    'POTENTIAL_ENERGY_RATING',
    'UPRN_SOURCE',
    'TENURE',
    'BUILT_FORM',
'TRANSACTION_TYPE',
'CONSTRUCTION_AGE_BAND',
]
    

def generate_mean_median_attribute_for_numeric_col(pc, data, col):
    # get count nulls 
    nul_count = data[col].isna().sum()
    # Calculate mean for the specified column
    try:
        # Handle potential NaN/null values by dropping them
        mean_value = data[col].dropna().mean()
        median_value = data[col].dropna().median()
        
        # Create dictionary with results
        results = {
            f"mean_{col}": round(mean_value, 3),  
             f"median_{col}": round(median_value, 3),
            'postcode': pc , 
            f'null_count_{col}': nul_count
        }
        
        return results
        
    except (TypeError, ValueError) as e:
        # Handle cases where column might not be numeric
        print(f"Error calculating mean for column {col}: {str(e)}")
        return {
            f"mean_{col}": None,
            f"median_{col}": None,
            f'null_count_{col}': nul_count,
            'postcode': pc
        }


        
    except (TypeError, ValueError) as e:
        # Handle cases where column might not be numeric
        print(f"Error calculating median for column {col}: {str(e)}")
        return {
            f"median_{col}": None,
            'postcode': pc
        }
    

def generate_percentage_atr(pc, data, col):
    # Calculate total number of buildings
    count_builds = data.shape[0]
    
    # Calculate percentages
    results = (data.groupby([col])['LMK_KEY'].count() / count_builds * 100).to_dict()
    
    # Create new dictionary with modified keys
    updated_results = {f"perc_{col}_{key}": value for key, value in results.items()}
    
    # Add postcode
    updated_results['postcode'] = pc
    
    return updated_results


def generate_sum_attribute(pc, data, col):
    #  error if there are nulls 
    
    # log count of null 
    nul_count = data[col].isna().sum()

    # Calculate sum for the specified column
    data[col] = data[col].fillna(data[col].mean())
    sum_value = data[col].sum() 
    # Create dictionary with results
    results = {
        f"sum_{col}": round(sum_value, 2),  # Round to 2 decimal places
        'postcode': pc,
        f'null_count_{col}': nul_count
    }
    return results


def process_postcode_attributes(data, postcode):
    # Filter dataframe for specific postcode
    
    # check if data not empty 
    if data.shape[0] == 0:
        print(f"No data found, : {postcode}")
        return None
    
    postcode_data = data[data['POSTCODE'] == postcode].copy()
    
    if postcode_data.empty:
        print(f"No data found for postcode: {postcode}")
        return None
    
    # Define numeric and categorical columns

    # Initialize results dictionary
    all_attributes = {}

    all_attributes['count_buildings'] = len(postcode_data)

    count_uprn = len(postcode_data.UPRN.unique().tolist() )
    count_building_ref = len(postcode_data.BUILDING_REFERENCE_NUMBER.unique().tolist() )
    
    all_attributes['count_uprn'] = count_uprn
    all_attributes['count_building_ref'] = count_building_ref

    # Process numeric columns
    for col in numeric_cols:
            mean_results = generate_mean_median_attribute_for_numeric_col(postcode, postcode_data, col)
            all_attributes.update(mean_results)
            
    
    # Process categorical columns
    for col in categorical_cols:
            percentage_results = generate_percentage_atr(postcode, postcode_data, col)
            all_attributes.update(percentage_results)
    
    for col in sum_cols:
            sum_results = generate_sum_attribute(postcode, postcode_data, col)
            all_attributes.update(sum_results)
    
    return all_attributes

# Example usage:
def process_multiple_postcodes(data, postcodes):
    all_results = []
    for pc in postcodes:
        results = process_postcode_attributes(data, pc)
        if results:
            all_results.append(results)
    
    return all_results


def analyse_epc(df, lad_code ):
    """ Takes an input df and calc all variables for all postcodes 
    """
 
    postcodes = df.POSTCODE.unique().tolist()

    # check for non nulls 
    if df.INSPECTION_DATE.isna().sum() != 0:
        print(df.INSPECTION_DATE.isna().sum() )
        if df.INSPECTION_DATE.isna().sum() <5:
            or_len = df.shape[0]
            print('Less than 5 nulls, removing rows ')
            df = df.dropna(subset=['INSPECTION_DATE'])
            df.reset_index(inplace=True)
            print(f"Removed {or_len - df.shape[0]} rows")
            
            assert df.INSPECTION_DATE.isna().sum() == 0
            
        else:
             raise ValueError('INSPECTION_DATE has nulls')

    if df.BUILDING_REFERENCE_NUMBER.isna().sum() != 0:
        raise ValueError('BUILDING_REFERENCE_NUMBER has nulls')  

    df['INSPECTION_DATE'] = pd.to_datetime(df['INSPECTION_DATE'])   
    latest_entries = df.iloc[df.groupby('BUILDING_REFERENCE_NUMBER')['INSPECTION_DATE'].agg(pd.Series.idxmax)].copy()
    latest_entries   = pre_process_cv_data(latest_entries)
    # checl not empty 
    assert latest_entries.shape[0] > 0
    # check no dups on uprn
    assert latest_entries.BUILDING_REFERENCE_NUMBER.duplicated().sum() == 0

    results = process_multiple_postcodes(latest_entries, postcodes)
    res_df = pd.DataFrame(results) 
    res_df['lad_code'] = lad_code   
    return res_df   

def pre_process_cv_data(df):
    
    for col in ['CO2_EMISS_CURR_PER_FLOOR_AREA', 'CO2_EMISSIONS_POTENTIAL', 'CO2_EMISSIONS_CURRENT',  'ENERGY_CONSUMPTION_POTENTIAL', 'ENERGY_CONSUMPTION_CURRENT']:
         df[col].mask(df[col] < 0, inplace=True)
    
    
    df['energy_consumption_total'] = df['ENERGY_CONSUMPTION_CURRENT'] * df['TOTAL_FLOOR_AREA']
    df['potential_energy_consumption_total'] = df['ENERGY_CONSUMPTION_POTENTIAL'] * df['TOTAL_FLOOR_AREA']  

    for col in categorical_cols:
        # fill nas with unknown 
        df[col] = df[col].fillna('unknown')
    return df
    
