import os
import glob
import math
import re
import pandas as pd
import numpy as np
from numba import njit
from dask import delayed, compute
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from tqdm.notebook import tqdm
import dask
dask.config.set(scheduler='threads')

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER

FINAL_FILE = folder_location.ABNORMAL_CSV

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
            print("========== Required labelled files not present ==========")
            print("=============== Begin label creation ===============")
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
        pass
    
    def __calculate_scores(self):
        '''
        calculates scores for each stock
        refer to data/calc_anomaly_scores.py for more details
        '''
        pass
    
    def __run_snorkel(self):
        '''
        runs snorkel to create labels from the scores
        '''
        pass
    