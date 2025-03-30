import pandas as pd
import numpy as np
import os

PROCESSED_DATA_FOLDER = "../data_untracked/processed"
ABNORMAL_CSV = "snorkel_labels.csv"

FINAL_FOLDER = "../data_untracked/features"
FINAL_FILE = "transaction_code.csv"

def create_features():
    """This function will create the key-feature footnote if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FINAL_FOLDER)
    # print(current_compiled_files)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Footnote Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
    else: # Create features and save
        print("=== Footnote Key file not found, begin creating  ===")
        
        ## Extract snorkel labels
        abnormal_transactions = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{ABNORMAL_CSV}')[["ACCESSION_NUMBER", "TRANS_SK","TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "snorkel_prob", "snorkel_pred"]]
        abnormal_transactions = abnormal_transactions.rename(columns={"snorkel_prob" : "probability", "snorkel_pred" : "prediction"})
        df_features = abnormal_transactions.copy()

        ##############################
        # Sell features
        ##############################
        ## Create a binary variable to extract Transcode = J and Trans Acquired = D which means J code and sell
        df_features["js_bin"] = np.where((df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D") & (df_features["TRANS_CODE"].str.upper() == "J"), 1, 0)

        ## Create a binary variable to extract Transcode Not J or S but Trans Acquired = D which is sell but non S or J coded
        df_features["os_bin"] = np.where((~df_features["TRANS_CODE"].str.upper().isin(["S", "J"])) & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "D"), 1, 0)
        ##############################
        # Buy features
        ##############################

        ## Create a binary variable to extract Transcode = P which is Buy and Acquired disp cd = A (Regular Sell)
        df_features["b_bin"] = np.where((df_features["TRANS_CODE"].str.upper() == "P") & (df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A"), 1, 0)

        ## Create a binary variable to extract Transcode = J and Trans Acquired = A which means J code and buy
        df_features["jb_bin"] = np.where((df_features["TRANS_ACQUIRED_DISP_CD"].str.upper() == "A") & (df_features["TRANS_CODE"].str.upper() == "J"), 1, 0)
        
        ##############################
        # Save file
        ##############################
        features_to_keep = ["js_bin", "b_bin", "jb_bin", "os_bin"]
        key = ["ACCESSION_NUMBER", "TRANS_SK"]

        df_to_save = df_features[features_to_keep + key]
        df_to_save.to_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save

    return df_to_return