import numpy as np
import pandas as pd
import os

from create_data import transaction_code_feature
from create_data import graph_feature
from create_data import footnote_feature
from create_data import other_feature

PROCESSED_DATA_FOLDER = '../data_untracked/processed'

FEATURES_DATA_FOLDER = '../data_untracked/features'

FINAL_FILE = 'snorkel_labels.csv'

class Feature_Data_Creator:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}')
    
    def create_features(self):
        ## Creates transaction code features
        self._create_transaction_code()
        
        ## Creates footnotes features
        self.__create_footnote_feature()
        
        ## Create graph features
        self.__create_graph_features()
        
        ## Create other features
        self.__create_other_features()

################################################################################
# Create transaction code
################################################################################

    def _create_transaction_code(self):
        
        key_columns = ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD"]
        feature_columns = ["js_bin", "b_bin", "jb_bin", "os_bin"]
        
        data_to_merge = transaction_code_feature.create_features()
        
        self.__merge_features(data_to_merge, key_columns, feature_columns)
        
        

################################################################################
# Create Footnote features
################################################################################

    def __create_footnote_feature(self):
        ## Code for label creation
        pass
    
################################################################################
# Create Graph features
################################################################################

    def __create_graph_features(self):
        ## Code for label creation
        pass


################################################################################
# Create Other features
################################################################################

    def __create_other_features(self):
        ## Code for label creation
        pass
    

################################################################################
# Auto merge features
################################################################################  

    def __merge_features(self, data_to_merge, key_columns, feature_columns):
        
        data = pd.merge(
            self.data,
            data_to_merge,
            left_on = key_columns,
            right_on = key_columns,
            how = "left"
        )
        
        data[feature_columns] = data[feature_columns].fillna(0)
        
        
        self.data = data