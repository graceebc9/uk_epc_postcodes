import os 
from src import run_gmm_clustering 
import pandas as pd 
from src_clustering import convert_residuals_into_full_df ,convert_residuals_to_bands
import pandas as pd

def main(neb_data_path, pc_path, model_predictions_path, output_dir, run_name , cluster_columns,  test=False  ): 
        
    res = pd.read_csv(model_predictions_path) 
    # pc_path='/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998'

    neb = pd.read_csv(neb_data_path ) 

    res_geo = convert_residuals_into_full_df(res, neb, pc_path ) 

    res_df = convert_residuals_to_bands(res_geo )

    # opone hot encode prediction_band
    res_df = pd.get_dummies(res_df, columns=['prediction_band'])
    
   

    type_cols = [  'Domestic outbuilding_pct',
    'Large detached_pct',
    'Large semi detached_pct',
    'Linked and step linked premises_pct',
    'Medium height flats 5-6 storeys_pct',
    'Planned balanced mixed estates_pct',
    'Semi type house in multiples_pct',
    'Small low terraces_pct',
    'Standard size detached_pct',
    'Standard size semi detached_pct',
    'Tall flats 6-15 storeys_pct',
    'Tall terraces 3-4 storeys_pct',
    'Very large detached_pct',
    'Very tall point block flats_pct',
    ]  

    # fillan for type cols 
    for col in type_cols:
        res_df[col] = res_df[col].fillna(0) 

    if test:
        cluster_data = res_df.iloc[0:1000]
    else:
        cluster_data = res_df   

    # store cluster columns in config file we save 
    # we want fn to save config of clustering 
    with open(os.path.join(output_dir, run_name, 'cluster_cols_config.txt') , 'w') as f:
        cluster_cols_save = '\n'.join(cluster_cols)
        f.write(cluster_cols_save)

    run_gmm_clustering(cluster_data, 
        cluster_columns, 
        input_name = 'postcode', 
        run_name=run_name , 
        output_dir=output_dir, 
        n_components_range=range(3, 15),
        exclude_lads=None,
    )


if __name__ == '__main__':

    cluster_cols = [  'Domestic outbuilding_pct',
    'Large detached_pct',
    'Large semi detached_pct',
    'Linked and step linked premises_pct',
    'Medium height flats 5-6 storeys_pct',
    'Planned balanced mixed estates_pct',
    'Semi type house in multiples_pct',
    'Small low terraces_pct',
    'Standard size detached_pct',
    'Standard size semi detached_pct',
    'Tall flats 6-15 storeys_pct',
    'Tall terraces 3-4 storeys_pct',
    'Very large detached_pct',
    'Very tall point block flats_pct',
    'percentage_residuals', 
    'all_types_total_buildings',
    ]  

    neb_data_path = '/home/gb669/rds/hpc-work/energy_map/data/automl_models/input_data/new_final/NEBULA_englandwales_domestic_filtered.csv' 
    pc_path = '/home/gb669/rds/hpc-work/energy_map/data/postcode_polygons/codepoint-poly_5267291' 
    model_predictions_path = '/home/gb669/rds/hpc-work/energy_map/data/automl_models/input_data/nebula_prediction_model18/predictions_all_neb.csv'      
    output_dir = '/home/gb669/rds/hpc-work/energy_map/uk_epc_postcodes/results/postcode_clustering'    
    run_name = 'neb18_full_run_v1'

    main(neb_data_path, pc_path, model_predictions_path, output_dir, run_name , cluster_cols,  test=False ) 