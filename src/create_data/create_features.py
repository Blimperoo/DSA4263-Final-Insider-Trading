import numpy as np
import pandas as pd
import os

from create_data import transaction_code_feature
from create_data import graph_feature
from create_data import footnote_feature
from create_data import other_feature

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FEATURES_DATA_FOLDER = folder_location.FEATURES_DATA_FOLDER

FINAL_FEATURES_FILE = folder_location.FULL_FEATURES_FILE

FINAL_FILE = folder_location.ABNORMAL_CSV

class Feature_Data_Creator:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        self.features = []
    
    def create_features(self):
        ## If full features file exist:
        processed_folder = os.listdir(PROCESSED_DATA_FOLDER)
        
        if (FINAL_FEATURES_FILE in processed_folder):
            print("=== Final features file present ===")
            self.__load_data_frame()
        else:
            print("=== Final features file not found. Begin creating ===")
        
            ## Creates transaction code features
            self._create_transaction_code()
            
            ## Creates footnotes features
            self.__create_footnote_feature()
            
            ## Create graph features
            self.__create_graph_features()
            
            ## Create other features
            self.__create_other_features()
            
            print("=== Saving file ===")
            self.__save_data_frame()

################################################################################
# Create transaction code
################################################################################

    def _create_transaction_code(self):
        
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"]
        feature_columns = ['js_bin', 's_bin','b_bin', 'jb_bin', 'ob_bin', 'g_bin']
        
        data_to_merge = transaction_code_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
        
        
################################################################################
# Create Footnote features
################################################################################

    def __create_footnote_feature(self):
        key_columns = ["ACCESSION_NUMBER"]
        feature_columns = ['gift', 'distribution', 'charity', 'price',
                           'number', 'ball', 'pursuant', '10b5-1', '16b-3']
        
        data_to_merge = footnote_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
    
################################################################################
# Create Graph features
################################################################################

    def __create_graph_features(self):
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;"] # removed "TRANS_DATE"
        feature_columns = ["lobbyist_score_final", "total_senate_connections", "total_house_connections", "combined_seniority_score", "PI_combined_total"]
        
        data_to_merge = graph_feature.create_features()
        data_to_merge['TRANS_DATE'] = pd.to_datetime(data_to_merge['TRANS_DATE'])
        self.__merge_features(data_to_merge, key_columns, feature_columns)


################################################################################
# Create Other features
################################################################################

    def __create_other_features(self):
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"]
        feature_columns = ["net_trading_intensity", "net_trading_amt", "relative_trade_size_to_self", "relative_trade_size_to_others"]
        
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
        
################################################################################
# Save DF
################################################################################ 

    def __save_data_frame(self):
        """saves the data frame with a prefix: feature_
        """
        features_naming = list(map(lambda x: "feature_" + x, self.features))
        feature_name_mapping = dict(zip(self.features, features_naming))
        
        data_to_save = self.data.rename(columns=feature_name_mapping)
        data_to_save.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}')
    
################################################################################
# Load DF
################################################################################

    def __load_data_frame(self):
        """loads data frame and extract features with prefix: feature_
        """
        load_data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', parse_dates=['TRANS_DATE'])
        
        features = list(load_data.columns[load_data.columns.str.contains('feature_')])
        features_cleaned = [feature.replace('feature_', '') for feature in features]
        feature_name_mapping = dict(zip(features, features_cleaned))
        
        self.features.extend(features_cleaned)
        
        cleaned_data = load_data.rename(columns=feature_name_mapping)
        self.data = cleaned_data