import pandas as pd
import os
from pathlib import Path
import glob
from tqdm import tqdm
from src.agg_fns import analyse_epc

def process_epc_data(base_path, results_dir, force_reprocess=False):
    """
    Process EPC data files with progress tracking and skip already processed files
    
    Args:
        base_path: Base directory containing the data
        results_dir: Directory to save the results
        force_reprocess: If True, reprocess all files even if results exist
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    patterns = ['domestic-W0*', 'domestic-E0*']
    
    # First, get all authority directories
    authority_dirs = []
    for pattern in patterns:
        authority_dirs.extend(glob.glob(os.path.join(base_path, 'all-domestic-certificates', pattern)))
    
    # Create set of authorities that need processing
    if not force_reprocess:
        # Get list of already processed authorities from existing result files
        existing_results = {f.split('_results.csv')[0] for f in os.listdir(results_dir) 
                          if f.endswith('_results.csv')}
        authorities_to_process = [dir_path for dir_path in authority_dirs 
                                if dir_path.split('domestic-')[-1] not in existing_results]
        if not authorities_to_process:
            print("All authorities have already been processed. Use force_reprocess=True to reprocess.")
            return
    else:
        authorities_to_process = authority_dirs

    # Create master progress bar
    with tqdm(total=len(authorities_to_process), desc="Processing authorities") as pbar:
        for authority_dir in authorities_to_process:
            authority_code = authority_dir.split('domestic-')[-1]
            cert_path = os.path.join(authority_dir, 'certificates.csv')
            output_file = os.path.join(results_dir, f'{authority_code}_results.csv')
            
            if os.path.exists(cert_path):
                try:
                    # Update progress bar description
                    pbar.set_description(f"Processing {authority_code}")
                    
                    # Read and process the data
                    df = pd.read_csv(cert_path)
                    results_df = analyse_epc(df, authority_code)
                    
                    # Save individual results file
                    results_df.to_csv(output_file)
                    
                except Exception as e:
                    tqdm.write(f"Error processing {authority_code}: {str(e)}")
                
                finally:
                    # Update progress bar
                    pbar.update(1)
            else:
                tqdm.write(f"Skipping {authority_code} - certificates.csv not found")
                pbar.update(1)
    
    # Print summary
    processed_files = len([f for f in os.listdir(results_dir) if f.endswith('_results.csv')])
    total_authorities = len(authority_dirs)
    print(f"\nProcessing complete:")
    print(f"- Total authorities available: {total_authorities}")
    print(f"- Total authorities processed: {processed_files}")
    print(f"- Remaining unprocessed: {total_authorities - processed_files}")

# Usage
if __name__ == "__main__":
    base_path = '/Volumes/T9/2024_Data_downloads/2025_epc_database'
    results_dir = '/Volumes/T9/01_2025_EPC_POSTCODES'
    # Process only unprocessed files
    process_epc_data(base_path, results_dir, force_reprocess=False)
    
    # To reprocess all files:
    # process_epc_data(base_path, force_reprocess=True)