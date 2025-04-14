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

from path_location import folder_location
import create_features

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FEATURES_DATA_FOLDER = folder_location.FEATURES_DATA_FOLDER

FINAL_FEATURES_FILE = folder_location.FULL_FEATURES_FILE

TRAINING_FILE = folder_location.TRAINING_FULL_FEATURES_FILE

TESTING_FILE = folder_location.TESTING_FULL_FEATURES_FILE

FEATURES_PROCESSED = "full_features_processed.csv"

TRANSACTION_CODE_FEATURE = create_features.TRANSACTION_CODE_FEATURE
FOOTNOTE_FEATURE = create_features.FOOTNOTE_FEATURE
GRAPH_FEATURE = create_features.GRAPH_FEATURE
OTHER_FEATURE = create_features.OTHER_FEATURE
NETWORK_TIME_IND_FEATURE = create_features.NETWORK_TIME_IND_FEATURE
NETWORK_TIME_DEP_FEATURE = create_features.NETWORK_TIME_DEP_FEATURE


FEATURES = create_features.FEATURES
IMPORTANT_KEYS = create_features.IMPORTANT_KEYS

PROBABILITY = create_features.PROBABILITY
PREDICTION = create_features.PREDICTION



class Feature_Preprocessor:
    def __init__(self):
        self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FEATURES_FILE}', parse_dates=['TRANS_DATE'])
        self.initial_rows = self.data.shape[0]
        
        ## Combined features
        self.features = FEATURES
        
################################################################################
# Extract found features
################################################################################
    def extract(self, features=FEATURES, scale_data = True):
        """Extract found features in feature columns and then preprocess it. Then finally create training and testing
        """
        relevant_data = self.data.copy()
        found_features = []
        
        for column in relevant_data.columns:
            if column in features:
                found_features.append(column)
    
        for column in found_features:
            self.preprocess(column, scale_data)
        
################################################################################
# Preprocess features
################################################################################ 
    def preprocess(self, feature, scale_data):
        """Preprocess data. If object type then create one hot encoding
           If int or float type then remove fill na with 0, remove infinite and scale to 0 - 1
        Args:
            feature (str): column name
        """
        relevant_data = self.data.copy()
        print(f"preprocess {feature} with type {relevant_data[feature].dtypes}")
        
        # Check if one hot encoding is needed
        if feature == "TRANS_CODE" or relevant_data[feature].dtypes == object:
            data_to_replace = pd.get_dummies(relevant_data, columns=[feature], dtype=int)
        
        # If is int or float
        elif relevant_data[feature].dtypes == np.int64 or relevant_data[feature].dtypes == np.float64:
            data_to_replace = relevant_data.copy()
            
            # Check if there is infinite
            if sum(np.isinf(relevant_data[feature])) > 0:
                
                # Replace infinite with next highest value
                to_sort = relevant_data[feature].unique()
                to_sort.sort()
                next_largest = to_sort[-2]
                
                # next_highest = relevant_data[relevant_data[feature] < np.inf].max()
                relevant_data[feature] = relevant_data[feature].replace([np.inf], next_largest)
                data_to_replace = relevant_data.copy()
                
            # Check if there is NA
            if sum(np.isnan(relevant_data[feature])) > 0:
                # Replace with NA with 0
                relevant_data[feature] = relevant_data[feature].fillna(0)
                data_to_replace = relevant_data.copy()
            
            # Check data scaling
            if scale_data and (relevant_data[feature].max() != 1 or relevant_data[feature].min() != 0):
                # Scale values to 0 to 1
                min_val, max_val = relevant_data[feature].min(), relevant_data[feature].max()
                relevant_data[feature] = relevant_data[feature].apply(lambda x: (x - min_val)/ (max_val - min_val))
                data_to_replace = relevant_data.copy()

        self.data = data_to_replace

################################################################################
# Create training and testing data
################################################################################

    def create_training_testing(self, quantile = 0.80, split_days = 60):
        """ Creates training and testing split by transaction date based on quantile
        """
        
        print(f"=== Begin creating based on quantile: {quantile}")
        curr_data = self.data.copy()
        date_to_split = curr_data['TRANS_DATE'].quantile(quantile)
        date_to_split_high = date_to_split + pd.Timedelta(split_days, unit='D')
        data_to_split_low = date_to_split - pd.Timedelta(split_days, unit='D')
        
        training_data = curr_data[curr_data['TRANS_DATE'] <= data_to_split_low].drop(columns=["TRANS_DATE", "TRANS_CODE"])
        testing_data = curr_data[curr_data['TRANS_DATE'] >= date_to_split_high].drop(columns=["TRANS_DATE", "TRANS_CODE"])
        
        print("=== Saving Training and Testing ===")
        training_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{TRAINING_FILE}", index=False)
        testing_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{TESTING_FILE}", index=False)

################################################################################
# Create training and testing data for baseline model
################################################################################

    def baseline_create_training_testing(self, quantile = 0.80, split_days = 60):
        """ Creates baseline model training and testing split by transaction date based on quantile
        """
        
        print(f"=== Begin creating based on quantile: {quantile}")
        curr_data = self.data.copy()
        date_to_split = curr_data['TRANS_DATE'].quantile(quantile)
        date_to_split_high = date_to_split + pd.Timedelta(split_days, unit='D')
        data_to_split_low = date_to_split - pd.Timedelta(split_days, unit='D')
        
        training_data = curr_data[curr_data['TRANS_DATE'] <= data_to_split_low].drop(columns=["TRANS_DATE"])
        testing_data = curr_data[curr_data['TRANS_DATE'] >= date_to_split_high].drop(columns=["TRANS_DATE"])
        
        print("=== Saving baseline Training and Testing ===")
        
        BASELINE_TRAINING_FILE = "training_full_features_baseline.csv"
        BASELINE_TESTING_FILE = "testing_full_features_baseline.csv"
        training_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{BASELINE_TRAINING_FILE}", index=False)
        testing_data.to_csv(f"{PROCESSED_DATA_FOLDER}/{BASELINE_TESTING_FILE}", index=False)