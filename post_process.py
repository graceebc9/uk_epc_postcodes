import pandas as pd  
import glob 
import sys 
sys.path.append('../')
from src.post_process import post_process

path = '/Volumes/T9/01_2025_EPC_POSTCODES/*csv' 
finla = [] 
files = glob.glob(path)
for f in files:
    if 'master_' not in f:
        print(f)
    
        df = pd.read_csv(f)
    finla.append(df)

master = pd.concat(finla)
result, _  = post_process(master)
print("Number of rows before:", len(master))
print("Number of rows after:", len(result))
print("Number of duplicate postcodes:", len(master) - len(result))
result.to_csv('/Volumes/T9/01_2025_EPC_POSTCODES/master_results.csv', index=False)