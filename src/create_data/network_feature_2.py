import pandas as pd
import numpy as np
import os
import sys

from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from path_location import folder_location

FEATURES_FOLDER = folder_location.FEATURES_DATA_FOLDER
FINAL_FILE_1 = "network_features.csv"


def create_time_independent_features():
    """This function will create the key-feature csv if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE_1 in current_compiled_files:
        print("=== Network Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_1}')
    else: 
        ### To be filled
        print("=== Network Key file not found, begin creating  ===")
        df_to_return = pd.DataFrame()


    return df_to_return