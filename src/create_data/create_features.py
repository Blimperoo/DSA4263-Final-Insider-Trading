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

FINAL_FILE = folder_location.ABNORMAL_CSV

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
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        
        self.transaction_code_features = TRANSACTION_CODE_FEATURE
        self.footnote_features = FOOTNOTE_FEATURE
        self.graph_features = GRAPH_FEATURE
        self.other_features = OTHER_FEATURE
        self.network_time_ind_features = NETWORK_TIME_IND_FEATURE
        self.network_time_ind_features_2 = NETWORK_FEATURE
        
        ## Combined features
        self.features = FEATURES

################################################################################
# Create training and testing data
################################################################################

    def create_training_testing(self, quantile = 0.80):
        """ Creates training and testing split by transaction date based on quantile
        """
        features_folder = os.listdir(PROCESSED_DATA_FOLDER)
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', parse_dates=['TRANS_DATE'])
        
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
            #self.__create_graph_features()

            ## Create network features
            self.__create_network_features()
            
            self.__create_network_features_2()
            
            ## Create other features
            self.__create_other_features()
            
            print("=== Saving file ===")
            self.__save_data()
            # self.data.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}')

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
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;"] 
        feature_columns = self.graph_features
        
        data_to_merge = graph_feature.create_features()
        data_to_merge['TRANS_DATE'] = pd.to_datetime(data_to_merge['TRANS_DATE'])
        self.__merge_features(data_to_merge, key_columns, feature_columns)

################################################################################
# Create Network features (Graph features remodelled)
################################################################################

    def __create_network_features(self):
        
        # First add time_independent_features
        key_columns = ["RPTOWNERCIK_;"] 
        time_ind_features = self.network_time_ind_features
        
        data_to_merge = network_feature.create_time_independent_features()
        self.__merge_features(data_to_merge, key_columns, time_ind_features)

        # Second add time_dependent_features
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"] 
        #time_dep_features = self.network_time_dep_features

        #data_to_merge = network_feature.create_time_dependent_features()
        #self.__merge_features(data_to_merge, key_columns, time_dep_features)
        
    def __create_network_features_2(self):
        
        # First add time_independent_features
        key_columns = ["TRANS_SK"] 
        time_ind_features = self.network_time_ind_features_2
        
        data_to_merge = network_feature_2.create_time_independent_features()
        self.__merge_features(data_to_merge, key_columns, time_ind_features)

        # Second add time_dependent_features
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"] 
        #time_dep_features = self.network_time_dep_features

        #data_to_merge = network_feature.create_time_dependent_features()
        #self.__merge_features(data_to_merge, key_columns, time_dep_features)


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

        # EMILY TO REMOVE HARD CODED CHANGE AFTER ADDING SNORKEL LABEL PARTS
        ################################################################################ 
        if "RPTOWNERCIK" in self.data.columns and 'RPTOWNERCIK_;' in key_columns:
            print("column is renamed. PLEASE CHANGE THIS ASAP")
            self.data.rename(columns={'RPTOWNERCIK': 'RPTOWNERCIK_;'}, inplace=True)
        
        print(data_to_merge)

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
# Save features + keys + labels
################################################################################  

    def __save_data(self):
        list_of_savable_features = []
        create_data = self.data.copy()
        
        columns_to_save = IMPORTANT_KEYS + FEATURES + PROBABILITY + PREDICTION
        
        columns = create_data.columns
        
        for column in columns:
            if column in columns_to_save:
                list_of_savable_features.append(column)
        
        create_data = create_data[list_of_savable_features]
        create_data.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}')
        self.data = create_data
        
x =Feature_Data_Creator()
x.create_training_testing()