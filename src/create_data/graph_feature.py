import pandas as pd
import numpy as np
import os

PROCESSED_DATA_FOLDER = "../data_untracked/processed"
ABNORMAL_CSV = "snorkel_labels.csv"

FINAL_FOLDER = "../data_untracked/features"
FINAL_FILE = "graph_feature.csv"

def create_features():
    """This function will create the key-feature graph if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FINAL_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Graph Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
    else: # Create features and save
        print("=== Graph Key file not found, begin creating  ===")
        
        
        
        """          Enter code here
        
        """
        
        
        df_to_save = ""
        df_to_save.to_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save

    return df_to_return