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
import network_feature
import network_feature_2
import footnote_feature
import other_feature

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FEATURES_DATA_FOLDER = folder_location.FEATURES_DATA_FOLDER

FINAL_FEATURES_FILE = folder_location.FULL_FEATURES_FILE

TRAINING_FILE = folder_location.TRAINING_FULL_FEATURES_FILE

TESTING_FILE = folder_location.TESTING_FULL_FEATURES_FILE

TRANSACTION_CODE_FEATURE = ['js_bin', 's_bin','b_bin', 'jb_bin', 'ob_bin', 'g_bin']
FOOTNOTE_FEATURE = ['gift', 'distribution', 'charity', 'price', 'number', 'ball', 'pursuant', '10b5-1', '16b-3']
GRAPH_FEATURE = ['lobbyist_score_final', 'total_senate_connections', 'total_house_connections', 'combined_seniority_score', 'PI_combined_total']
OTHER_FEATURE = ['net_trading_intensity', 'net_trading_amt', 'relative_trade_size_to_self', 'relative_trade_size_to_others','beneficial_ownership_score']
NETWORK_TIME_IND_FEATURE = ['is_lobby', 'has_lobby', 'has_donate']
NETWORK_FEATURE = ['important_connections',	'full_congress_connections', 'sen_important_connections', 'sen_full_congress_connections',
                   'sen_t2_full_congress_connections', 'sen_t1_important_connections', 'sen_t1_full_congress_connections',	'house_t2_important_connections',
                   'house_t2_full_congress_connections', 'house_t1_important_connections', 'house_t1_full_congress_connections']

# NETWORK_TIME_DEP_FEATURE = ['subcomm']
FEATURES = TRANSACTION_CODE_FEATURE + FOOTNOTE_FEATURE + GRAPH_FEATURE + OTHER_FEATURE + NETWORK_TIME_IND_FEATURE + NETWORK_FEATURE
IMPORTANT_KEYS = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;", "TRANS_DATE"]

PROBABILITY = ['snorkel_prob']
PREDICTION = ['snorkel_pred']

class Feature_Data_Creator:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        
        ## Combined features
        self.features = FEATURES
        
    def extract(self):
        relevant_data = self.data.copy()
        
        found_features = []
        
        for column in relevant_data.columns:
            if column in FEATURES:
                found_features.append(column)
        
        relevant_data = relevant_data[found_features + PROBABILITY + PREDICTION]
        self.data = relevant_data
        
        for column in found_features:
            # if column != "relative_trade_size_to_self":
            #     continue
            self.preprocess(column)
        
        self.data.to_csv("check1.csv")
        
    def preprocess(self, feature):
        
        
        relevant_data = self.data.copy()
        print(relevant_data[feature].dtypes, feature)
        
        # Check if one hot encoding is needed
        if relevant_data[feature].dtypes == object:
            data_to_replace = pd.get_dummies(relevant_data, columns=[feature], dtype=int)
        
        
        elif relevant_data[feature].dtypes == np.int64 or relevant_data[feature].dtypes == np.float64:
            print("yep")
            data_to_replace = relevant_data.copy()
            
            # Check if there is infinite
            if sum(np.isinf(relevant_data[feature])) > 0:
                # max_val = relevant_data[feature].max()
                print(relevant_data[feature].dtypes)
                to_sort = relevant_data[feature].unique()
                to_sort.sort()
                next_largest = to_sort[-2]
                
                # next_highest = relevant_data[relevant_data[feature] < np.inf].max()
                relevant_data[feature] = relevant_data[feature].replace([np.inf], next_largest)
                data_to_replace = relevant_data.copy()
                
            # Check if there is NA
            if sum(np.isnan(relevant_data[feature])) > 0:
                relevant_data[feature] = relevant_data[feature].fillna(0)
                data_to_replace = relevant_data.copy()
            
            # Check data scaling
            if (relevant_data[feature].max() != 1 or relevant_data[feature].min() != 0):
                min_val, max_val = relevant_data[feature].min(), relevant_data[feature].max()
                relevant_data[feature] = relevant_data[feature].apply(lambda x: (x - min_val)/ (max_val - min_val))
                data_to_replace = relevant_data.copy()

        self.data = data_to_replace
    
        
        
                
Feature_Data_Creator().extract()
        