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
FINAL_FILE = "transaction_code.csv"

FEATURES_TO_KEEP = ['js_bin', 's_bin','b_bin', 'jb_bin', 'ob_bin', 'gb_bin']
KEY = ["ACCESSION_NUMBER", "TRANS_SK"]

def create_features():
    """This function will create the key-feature transaction key if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Transaction Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE}')
    else: # Create features and save
        print("=== Transaction Key file not found, begin creating  ===")
        
        ## Extract snorkel labels
        abnormal_transactions = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{TRANSACTIONS_LABELLED_FILE}')[["ACCESSION_NUMBER", "TRANS_SK", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD"]]
        df_features = abnormal_transactions.copy()
        
        ##############################
        # Sell
        ##############################

        ## Create a binary variable to extract Transcode = J and Trans Acquired = D which means J code and sell
        df_features["js_bin"] = np.where((df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D") & (df_features["TRANS_CODE"].str.upper() == "J"), 1, 0)

        ## Create a binary variable to extract Transcode Not J or S but Trans Acquired = D which is sell but non S or J coded
        df_features["os_bin"] = np.where((~df_features["TRANS_CODE"].str.upper().isin(["S", "J", "G"])) & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D"), 1, 0)

        ## Create a binary variable to extract Transcode = S which is sell and Acquired disp cd = D (Regular Sell)
        df_features["s_bin"] = np.where((df_features["TRANS_CODE"].str.upper() == "S") & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D" ), 1, 0)

        ##############################
        # Buy
        ##############################

        ## Create a binary variable to extract Transcode = P which is Buy and Acquired disp cd = A (Regular Sell)
        df_features["b_bin"] = np.where((df_features["TRANS_CODE"].str.upper() == "P") & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A"), 1, 0)

        ## Create a binary variable to extract Transcode = J and Trans Acquired = A which means J code and buy
        df_features["jb_bin"] = np.where((df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A") & (df_features["TRANS_CODE"].str.upper() == "J"), 1, 0)

        ## Create a binary variable to extract Transcode Not J or P but Trans Acquired = A which is buy but non P or J coded
        df_features["ob_bin"] = np.where((~df_features["TRANS_CODE"].str.upper().isin(["P", "J", "G"])) & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A"), 1, 0)

        ##############################
        # GIFTS
        ##############################

        ## Create a binary variable to extract if Transcode = G which is a gift or not
        df_features["gb_bin"] = np.where((df_features["TRANS_CODE"].str.upper() == "G") & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A"), 1, 0)
        
        ##############################
        # Save file
        ##############################
        df_to_save = df_features[FEATURES_TO_KEEP + KEY]
        df_to_save.to_csv(f'{FEATURES_FOLDER}/{FINAL_FILE}', index=False)
        df_to_return = df_to_save

    return df_to_return