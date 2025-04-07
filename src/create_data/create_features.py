import numpy as np
import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import transaction_code_feature
import graph_feature
import footnote_feature
import other_feature

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FEATURES_DATA_FOLDER = folder_location.FEATURES_DATA_FOLDER

FINAL_FEATURES_FILE = folder_location.FULL_FEATURES_FILE

FINAL_FILE = folder_location.ABNORMAL_CSV

TRAINING_FILE = folder_location.TRAINING_FULL_FEATURES_FILE

TESTING_FILE = folder_location.TESTING_FULL_FEATURES_FILE

class Feature_Data_Creator:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        
        self.transaction_code_features = ['js_bin', 's_bin','b_bin', 'jb_bin', 'ob_bin', 'g_bin']
        self.footnote_features = ['gift', 'distribution', 'charity', 'price', 'number', 'ball', 'pursuant', '10b5-1', '16b-3']
        self.graph_features = ['lobbyist_score_final', 'total_senate_connections', 'total_house_connections', 'combined_seniority_score', 'PI_combined_total']
        self.other_features = ['net_trading_intensity', 'net_trading_amt', 'relative_trade_size_to_self', 'relative_trade_size_to_others']
        
        ## Combined features
        self.features = self.transaction_code_features + self.footnote_features + self.graph_features + self.other_features

################################################################################
# Create training and testing data
################################################################################

    def create_training_testing(self, quantile = 0.70):
        """ Creates training and testing split by transaction date based on quantile
        """
        features_folder = os.listdir(PROCESSED_DATA_FOLDER)
        
        if (TRAINING_FILE not in features_folder) or (TESTING_FILE not in features_folder):
            print(f"=== Training or Testing file not found. Begin creating based on quantile: {quantile}")
            curr_data = self.data.copy()
            date_to_split = curr_data['TRANS_DATE'].quantile(quantile)
            
            training_data = curr_data[curr_data['TRANS_DATE'] < date_to_split]
            testing_data = curr_data[curr_data['TRANS_DATE'] >= date_to_split]
            
            print("=== Saving Training and Testing ===")
            training_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{TRAINING_FILE}")
            testing_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{TESTING_FILE}")
        else:
            print("=== Training and Testing file present ===")
            self.data.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}')


################################################################################
# Create features
################################################################################

    def create_features(self):
        """ Loads feature csv file if exists. Else start creating and saving
        """
        ## If full features file exist:
        processed_folder = os.listdir(PROCESSED_DATA_FOLDER)
        
        if (FINAL_FEATURES_FILE in processed_folder):
            print("=== Final features file present ===")
            load_data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', parse_dates=['TRANS_DATE'])
            self.data = load_data
        else:
            print("=== Final features file not found. Begin creating ===")
        
            ## Creates transaction code features
            self.__create_transaction_code_features()
            
            ## Creates footnotes features
            self.__create_footnote_features()
            
            ## Create graph features
            self.__create_graph_features()
            
            ## Create other features
            self.__create_other_features()
            
            print("=== Saving file ===")
            self.data.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}')

################################################################################
# Create transaction code features
################################################################################

    def __create_transaction_code_features(self):
        
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"]
        feature_columns = self.transaction_code_features
        
        data_to_merge = transaction_code_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
        
        
################################################################################
# Create Footnote features
################################################################################

    def __create_footnote_features(self):
        key_columns = ["ACCESSION_NUMBER"]
        feature_columns = self.footnote_features
        
        data_to_merge = footnote_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
    
################################################################################
# Create Graph features
################################################################################

    def __create_graph_features(self):
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;"] # removed "TRANS_DATE"
        feature_columns = self.graph_features
        
        data_to_merge = graph_feature.create_features()
        data_to_merge['TRANS_DATE'] = pd.to_datetime(data_to_merge['TRANS_DATE'])
        self.__merge_features(data_to_merge, key_columns, feature_columns)


################################################################################
# Create Other features
################################################################################

    def __create_other_features(self):
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"]
        feature_columns = self.other_features
        
        data_to_merge = other_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
    

################################################################################
# Auto merge features
################################################################################  

    def __merge_features(self, data_to_merge, key_columns, feature_columns):
        data = pd.merge(
            self.data,
            data_to_merge[key_columns + feature_columns],
            on = key_columns,
            how = "left"
        )
        
        if data.shape[0] != self.initial_rows:
            print("Rows mismatch after merging, new, old: ", data.shape[0], self.initial_rows)
        
        self.features.extend(feature_columns)
        
        self.data = data
