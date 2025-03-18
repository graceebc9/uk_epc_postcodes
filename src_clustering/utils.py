# import pandas as pd


# import os 
# import pandas as pd 
# import geopandas as gpd 

# def load_pc_shp(pcs_to_load, pc_path='/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998'):
#     ll = []
#     for pc in pcs_to_load:
#         if len(pc)==1:
#             path = os.path.join(pc_path, f'codepoint-poly_5267291/one_letter_pc_code/{pc}/{pc}.shp')
#         else:
#             path = os.path.join(pc_path,  f'codepoint-poly_5267291/two_letter_pc_code/{pc}.shp' ) 
#         sd = gpd.read_file(path)    
#         ll.append(sd) 
#     pc_shp = pd.concat(ll)
#     return pc_shp 


# def create_geo_df(df, pc_path='/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998', hpc = False):
#     pcs_load = df.postcode.str[0:2].unique().tolist()
#     pcs_load = df.postcode.str.extract('([A-Z]+)').iloc[:,0].unique().tolist()
#     if hpc:
#         pcs_load= [x.lower() for x in pcs_load]
#     print('Postcodes to load ' , pcs_load)
#     pc_shp = load_pc_shp(pcs_load, pc_path)
    
#     geo_df = pc_shp.merge(df, left_on='POSTCODE', right_on='postcode', how='inner')
#     geo_df['latitude'] = geo_df.geometry.centroid.y 
#     geo_df['longitude'] = geo_df.geometry.centroid.x
#     geo_df = geo_df.to_crs('EPSG:4326')


#     return geo_df

# import pandas as pd


# def convert_residuals_into_full_df(yh, neb, pc_path, hpc = False ): 
#     print(yh.shape)
#     yh['residuals'] = yh['total_gas'] - yh['predictions']
#     yh['percentage_residuals'] = yh['residuals'] / yh['total_gas']
#     yh = pd.merge(yh, neb, on='postcode', how='inner')
#     print(yh.shape)
#     yh_geo = create_geo_df(yh, pc_path, hpc=hpc )
#     return yh_geo


# import numpy as np
# import pandas as pd

# def convert_residuals_to_bands(df, col='percentage_residuals', n=3):
#     """
#     Convert percentage residuals into bands for over and under predicting.
    
#     Parameters:
#     - df: DataFrame containing the residuals
#     - col: Name of the column containing percentage residuals
#     - n: Number of quantile bands to create
    
#     Returns:
#     - DataFrame with additional 'prediction_band' column
#     """
#     # Create a copy to avoid modifying the original DataFrame
#     result_df = df.copy()
    
#     # Create a new column for the direction
#     result_df['pred_direction'] = np.where(result_df[col] < 0, 'OVER', 
#                                    np.where(result_df[col] > 0, 'UNDER', 'EXACT'))
    
#     # Initialize prediction_band as object type to avoid categorical issues
#     result_df['prediction_band'] = None
    
#     # Handle each group separately to assign bands
#     for direction in ['OVER', 'UNDER']:
#         mask = result_df['pred_direction'] == direction
#         if mask.sum() >= n:  # Only apply qcut if we have enough data points
#             # For over-prediction, use absolute values to keep ordering intuitive
#             values = result_df.loc[mask, col].abs() if direction == 'OVER' else result_df.loc[mask, col]
            
#             # Convert to regular strings instead of categories
#             labels = [f'{direction}_{i+1}' for i in range(n)]
#             band_series = pd.qcut(values, n, labels=labels)
            
#             # Convert categorical to string before assignment
#             result_df.loc[mask, 'prediction_band'] = band_series.astype(str)
    
#     # Handle the exact predictions (residual = 0)
#     result_df.loc[result_df['pred_direction'] == 'EXACT', 'prediction_band'] = 'EXACT'
    
#     # Drop the temporary direction column
#     result_df.drop('pred_direction', axis=1, inplace=True)
    
#     return result_df


