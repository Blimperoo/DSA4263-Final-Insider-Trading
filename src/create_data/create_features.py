import numpy as np
import pandas as pd
import dask.dataframe as dd
import sys
import os
import gc

gc.enable()

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import transaction_code_feature
import graph_feature
import network_feature
import footnote_feature
import pagerank_feature
import other_feature

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FEATURES_DATA_FOLDER = folder_location.FEATURES_DATA_FOLDER

FINAL_FEATURES_FILE = folder_location.FULL_FEATURES_FILE

FINAL_FILE = folder_location.TRANSACTIONS_LABELLED_FILE

TRAINING_FILE = folder_location.TRAINING_FULL_FEATURES_FILE

TESTING_FILE = folder_location.TESTING_FULL_FEATURES_FILE

TRANSACTION_CODE_FEATURE = ['js_bin', 's_bin','b_bin', 'jb_bin', 'ob_bin', 'gb_bin']
FOOTNOTE_FEATURE = ['gift', 'distribution', 'charity', 'price', 'number', 'ball', 'pursuant', '10b5-1', '16b-3']
GRAPH_FEATURE = ['lobbyist_score_final', 'total_senate_connections', 'total_house_connections', 'combined_seniority_score', 'PI_combined_total']

OTHER_FEATURE = ['net_trading_intensity', 'net_trading_amt', 'relative_trade_size_to_self', 'beneficial_ownership_score','title_score',
                 'TRANS_TIMELINESS_clean', 'execution_timeliness', 'filing_lag_days', 'filing_timeliness']

NETWORK_TIME_IND_FEATURE = ['is_lobby', 'has_lobby', 'has_donate', 'NODEID']

NETWORK_TIME_DEP_FEATURE = ['important_connections', 'full_congress_connections', 
                            'house_t2_important_connections', 'house_t2_full_congress_connections', 
                            'house_t1_important_connections', 'house_t1_full_congress_connections'
                            'sen_important_connections', 'sen_full_congress_connections', 
                            'sen_t2_important_connections', 'sen_t2_full_congress_connections', 
                            'sen_t1_important_connections', 'sen_t1_full_congress_connections']

NETWORK_ZSCORE_FEATURE = ['full_congress_connections_z', 'sen_full_congress_connections_z', 'sen_t2_full_congress_connections_z','house_t2_full_congress_connections_z', 
                          'sen_important_connections_z', 'sen_t2_important_connections_z','important_connections_z', 'house_t2_important_connections_z', 
                          'full_congress_connections_z_cat', 'full_congress_connections_z_is_low', 'full_congress_connections_z_is_high','sen_full_congress_connections_z_cat',
                          'sen_full_congress_connections_z_is_low', 'sen_full_congress_connections_z_is_high', 'sen_t2_full_congress_connections_z_cat','sen_t2_full_congress_connections_z_is_low',
                          'sen_t2_full_congress_connections_z_is_high','house_t2_full_congress_connections_z_cat', 'house_t2_full_congress_connections_z_is_low', 'house_t2_full_congress_connections_z_is_high',
                          'sen_important_connections_z_cat', 'sen_important_connections_z_is_low', 'sen_important_connections_z_is_high', 'sen_t2_important_connections_z_cat',
                          'sen_t2_important_connections_z_is_low', 'sen_t2_important_connections_z_is_high', 'important_connections_z_cat', 'important_connections_z_is_low',
                          'important_connections_z_is_high', 'house_t2_important_connections_z_cat', 'house_t2_important_connections_z_is_low','house_t2_important_connections_z_is_high']

PAGERANK_FEATURE = ['ppr_topK_exp', 'num_topK_neighbors', 'ppr_house_0.85', 'ppr_house_0.95', 'ppr_senate_0.85', 'ppr_senate_0.95']



FEATURES = TRANSACTION_CODE_FEATURE + FOOTNOTE_FEATURE + GRAPH_FEATURE + OTHER_FEATURE + \
            NETWORK_TIME_IND_FEATURE + NETWORK_TIME_DEP_FEATURE + NETWORK_ZSCORE_FEATURE + PAGERANK_FEATURE


IMPORTANT_KEYS = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;", "TRANS_CODE"]
PROBABILITY = ['snorkel_prob']
PREDICTION = ['y_pred']

class Feature_Data_Creator:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        
        self.transaction_code_features = TRANSACTION_CODE_FEATURE
        self.footnote_features = FOOTNOTE_FEATURE
        self.graph_features = GRAPH_FEATURE
        self.other_features = OTHER_FEATURE
        self.network_time_ind_features = NETWORK_TIME_IND_FEATURE
        self.network_time_dep_features = NETWORK_TIME_DEP_FEATURE
        self.network_zscore_features = NETWORK_ZSCORE_FEATURE
        self.pagerank_features = PAGERANK_FEATURE
        
        ## Combined features
        self.features = FEATURES


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

            ## Create pagerank features
            self.__create_pagerank_features()
            
            ## Create network features
            self.__create_network_features()
        
            ## Creates transaction code features
            self.__create_transaction_code_features()
            
            ## Creates footnotes features
            self.__create_footnote_features()
            
            ## Create graph features
            self.__create_graph_features()
            
            ## Create other features
            self.__create_other_features()
            
            print("=== Removing unwanted rows ===")
            self.__remove_na_rows(["NODEID"])
            
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
        key_columns = ["RPTOWNERCIK"] 
        time_ind_features = self.network_time_ind_features
        
        data_to_merge = network_feature.create_time_independent_features()
        self.__merge_features(data_to_merge, key_columns, time_ind_features)

        # Second add time_dependent_features
        key_columns = ["ACCESSION_NUMBER", "TRANS_SK"] 
        time_dep_features = self.network_time_dep_features

        data_to_merge = network_feature.create_time_dependent_features()
        self.__merge_features(data_to_merge, key_columns, time_dep_features)
        
        # Third add network_zscore_feature
        key_columns = ["ACCESSION_NUMBER"] 
        network_zscore_feature = self.network_zscore_features

        data_to_merge = network_feature.create_zscore_features()
        self.__merge_features(data_to_merge, key_columns, network_zscore_feature)

################################################################################
# Create pagerank features
################################################################################

    def __create_pagerank_features(self):
        key_columns = ["ACCESSION_NUMBER"] 
        feature_columns = self.pagerank_features
        
        data_to_merge = pagerank_feature.create_features()
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
        
        # if len(feature_columns) > 15:
        #     data = dd.merge(self.data,
        #                   data_to_merge,
        #                   on = key_columns, 
        #                   how = "left")
        # else:

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
# Remove unwanted rows
################################################################################

    def __remove_na_rows(self, columns):
        to_remove = self.data.copy()
        # to_remove.to_csv("check2.csv", index=False)
        
        print(f"=== Before removal length {len(to_remove)} === ")
        to_remove = to_remove.dropna(subset=columns)
        to_remove = to_remove.drop(columns=columns)
        print(f"=== After removal length {len(to_remove)} === ")
        
        self.data = to_remove

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
        create_data.to_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', index=False)
        self.data = create_data
