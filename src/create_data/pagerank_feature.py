import pandas as pd
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER
TRANSACTIONS_LABELLED_FILE = folder_location.TRANSACTIONS_LABELLED_FILE

FEATURES_FOLDER = folder_location.FEATURES_DATA_FOLDER
FINAL_FILE = "pagerank_feature.csv"

def create_features():
    """This function will create the key-feature transaction key if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Pagerank Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE}')
    else: # Create features and save
        print("=== Pagerank Key file not found, begin creating  ===")
        df_to_return = pd.DataFrame()

    return df_to_return