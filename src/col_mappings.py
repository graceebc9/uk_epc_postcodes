# to do check age whu is failing 

# age 

def map_to_meta_age_band(column_name):
    """
    Maps construction age band column names to simplified meta categories.
    
    Args:
        column_name (str): The original column name
        
    Returns:
        str: The meta age band category
    """
    # Strip prefix for cleaner processing
    cleaned_name = column_name.replace('perc_CONSTRUCTION_AGE_BAND_', '')
    
    # Handle special categories first
    if any(x in cleaned_name for x in ['unknown', 'NO DATA!', 'INVALID!']):
        return 'age_band_unknown'
    
    # Handle official England and Wales bands
    if 'England and Wales:' in cleaned_name:
        year_part = cleaned_name.split(':')[1].strip()
        
        # Map official bands
        if 'before 1900' in year_part:
            return 'perc_age_band_pre1900'
        elif '1900-1929' in year_part:
            return 'perc_age_band_1900_1929'
        elif '1930-1949' in year_part:
            return 'perc_age_band_1930_1949'
        elif '1950-1966' in year_part:
            return 'perc_age_band_1950_1966'
        elif any(x in year_part for x in ['1967-1975', '1976-1982']):
            return 'perc_age_band_1967_1982'
        elif any(x in year_part for x in ['1983-1990', '1991-1995', '1996-2002']):
            return 'perc_age_band_1983_2002'
        elif any(x in year_part for x in ['2003-2006', '2007 onwards', '2007-2011', '2012 onwards']):
            return 'perc_age_band_2003_present'
    
    # Handle individual years
    try:
        year = int(''.join(filter(str.isdigit, cleaned_name)))
        
        if year < 1900:
            return 'perc_age_band_pre1900'
        elif 1900 <= year <= 1929:
            return 'perc_age_band_1900_1929'
        elif 1930 <= year <= 1949:
            return 'perc_age_band_1930_1949'
        elif 1950 <= year <= 1966:
            return 'perc_age_band_1950_1966'
        elif 1967 <= year <= 1982:
            return 'perc_age_band_1967_1982'
        elif 1983 <= year <= 2002:
            return 'perc_age_band_1983_2002'
        elif 2003 <= year: 
            return 'perc_age_band_2003_present'
      
    except ValueError:
        return 'perc_age_band_unknown'  # If we can't parse a year
    
    return 'perc_age_band_unknown'  # Fallback for any unhandled cases

def get_mapping_for_age_columns(columns):
    """
    Creates a dictionary mapping original column names to meta age bands.
    
    Args:
        columns (list): List of original column names
        
    Returns:
        dict: Mapping of original names to meta age bands
    """
    return {col: map_to_meta_age_band(col) for col in columns}


# tenure 

def map_to_tenure_category(column_name):
    """
    Maps tenure column names to standardized categories.
    
    Args:
        column_name (str): The original column name
        
    Returns:
        str: The standardized tenure category
    """
    # Strip prefix for cleaner processing
    cleaned_name = column_name.replace('perc_TENURE_', '').lower()
    
    # Handle owner-occupied variations
    if any(x in cleaned_name for x in ['owner-occupied', 'owner occupied']):
        return 'perc_tenure_owner_occupied'
    
    # Handle social rented variations
    if any(x in cleaned_name for x in ['rented (social)', 'rental (social)']):
        return 'perc_tenure_social_rented'
    
    # Handle private rented variations
    if any(x in cleaned_name for x in ['rented (private)', 'rental (private)']):
        return 'perc_tenure_private_rented'
    
    # Handle unknown and undefined cases
    if any(x in cleaned_name for x in ['unknown', 'not defined', 'no data']):
        return 'perc_tenure_unknown'
    
    # Fallback for any unhandled cases
    return 'perc_tenure_unknown'

def get_tenure_mapping_for_columns(columns):
    """
    Creates a dictionary mapping original column names to standardized tenure categories.
    
    Args:
        columns (list): List of original column names
        
    Returns:
        dict: Mapping of original names to standardized tenure categories
    """
    return {col: map_to_tenure_category(col) for col in columns}


# built form 


def map_to_built_form_category(column_name):
    """
    Maps built form column names to standardized categories while maintaining granularity.
    
    Args:
        column_name (str): The original column name
        
    Returns:
        str: The standardized built form category
    """
    # Strip prefix for cleaner processing
    cleaned_name = column_name.replace('perc_BUILT_FORM_', '').lower()
    
    # Handle each specific built form type
    if 'enclosed end-terrace' in cleaned_name:
        return 'perc_built_form_enclosed_end_terrace'
    elif 'enclosed mid-terrace' in cleaned_name:
        return 'perc_built_form_enclosed_mid_terrace'
    elif 'end-terrace' in cleaned_name:
        return 'perc_built_form_end_terrace'
    elif 'mid-terrace' in cleaned_name:
        return 'perc_built_form_mid_terrace'
    elif 'semi-detached' in cleaned_name:
        return 'perc_built_form_semi_detached'
    elif 'detached' in cleaned_name:
        return 'perc_built_form_detached'
    
    # Handle unknown and no data cases
    if any(x in cleaned_name for x in ['unknown', 'no data']):
        return 'perc_built_form_unknown'
    
    # Fallback for any unhandled cases
    return 'built_form_unknown'

def get_built_form_mapping_for_columns(columns):
    """
    Creates a dictionary mapping original column names to standardized built form categories.
    
    Args:
        columns (list): List of original column names
        
    Returns:
        dict: Mapping of original names to standardized built form categories
    """
    return {col: map_to_built_form_category(col) for col in columns}


# transaction type 


def map_to_transaction_type_category(column_name):
    """
    Maps transaction type column names to standardized categories.
    Returns categories with perc_ prefix.
    Any categories marked as 'do not use' or for backwards compatibility are mapped to unknown.
    """
    # Strip prefix for cleaner processing
    cleaned_name = column_name.replace('perc_TRANSACTION_TYPE_', '').lower()
    
    # If it's marked as "do not use" or "backwards compatibility", map to unknown
    if 'should not be used' in cleaned_name or 'backwards compatibility' in cleaned_name:
        return 'perc_transaction_unknown'
    
    # Handle sales categories - keeping marketed and non-marketed distinct
    if 'non marketed sale' in cleaned_name:
        return 'perc_transaction_non_marketed_sale'
    elif 'marketed sale' in cleaned_name:
        return 'perc_transaction_marketed_sale'

    
    # Handle rental categories
    if 'rental (social)' in cleaned_name and 'backwards compatibility' not in cleaned_name:
        return 'perc_transaction_rental_social'
    elif 'rental (private)' in cleaned_name and 'backwards compatibility' not in cleaned_name:
        return 'perc_transaction_rental_private'
    elif cleaned_name == 'rental':
        return 'perc_transaction_rental_general'
    
    # Handle assessment and application categories
    if 'fit application' in cleaned_name:
        return 'perc_transaction_fit_application'
    elif 'eco assessment' in cleaned_name:
        return 'perc_transaction_eco_assessment'
    elif 'rhi application' in cleaned_name:
        return 'perc_transaction_rhi_application'
    elif any(x in cleaned_name for x in ['assessment for green deal', 'following green deal']):
        return 'perc_transaction_green_deal'
    elif any(x in cleaned_name for x in ['stock condition survey']):
        return 'perc_transaction_stock_survey'
    
    # Handle new dwelling
    if 'new dwelling' in cleaned_name:
        return 'perc_transaction_new_dwelling'
    
    # Handle various "not" categories
    if any(x in cleaned_name for x in ['none of the above', 'not sale or rental']):
        return 'perc_transaction_other'
    
    # Handle unknown and not recorded cases
    if any(x in cleaned_name for x in ['unknown', 'not recorded', 'no data']):
        return 'perc_transaction_unknown'
    
    # Fallback for any unhandled cases
    return 'perc_transaction_unknown'

def get_transaction_type_mapping_for_columns(columns):
    """
    Creates a dictionary mapping original column names to standardized transaction type categories.
    """
    return {col: map_to_transaction_type_category(col) for col in columns}
