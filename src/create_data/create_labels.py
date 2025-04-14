import pandas as pd
import numpy as np
from numba import njit
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from tqdm.notebook import tqdm
import dask
import sys
import os
dask.config.set(scheduler='threads')

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FINAL_FILE = folder_location.TRANSACTIONS_LABELLED_FILE

class Label_Data_Creator:
    def __init__(self):
        self.data = None
    
    def create_labels(self):
        if self.__check_compiled_files():
            print("========== Required labelled files present ==========")
            print("========== Skip data label creation ==========")
            self.data = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{FINAL_FILE}')
            print("========== Reading successful ==========")
            
        else:
            print(f"========== Required labelled files not present. Please download from shared google drive: processed/{FINAL_FILE} ==========")
            print("========== Steps to reproduce. Please be weary of high computation resources required ==========")
            self.__calculate_ar_and_car()
            self.__calculate_scores()
            self.__run_snorkel()


################################################################################
# CHECK IF LABELLED FILE EXIST
################################################################################

    def __check_compiled_files(self):
        current_compiled_files = os.listdir(PROCESSED_DATA_FOLDER)
        return FINAL_FILE in current_compiled_files

################################################################################
# Create labels
################################################################################

    def __calculate_ar_and_car(self):
        '''
        calculates abnormal returns and cumulative abnormal returns for each stock
        refer to data/calc_abnormal_returns.py for more details
        '''
        print("---- 1. Calculate abnormal return and cumulative abnormal returns. See ../data/calc_abnormal_returns.ipynb ----")
        pass
    
    def __calculate_scores(self):
        '''
        calculates scores for each stock
        refer to data/calc_anomaly_scores.py for more details
        '''
        print("---- 2. Calculate anomaly scores for snorkel labelling. See ../data/calc_anomaly_scores.ipynb ----")
        pass
    
    def __run_snorkel(self):
        '''
        runs snorkel to create labels from the scores
        '''
        print("---- 3. Run snorkel labelling. See ../data/run_snorkel_labelling.ipynb ----")
        pass
    