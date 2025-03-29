import numpy as np
import pandas as pd
import os

import pandas as pd


PROCESSED_DATA_FOLDER = '../data_untracked/processed'

FINAL_FILE = 'snorkel_labels.csv'

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
            print("========== Required labelled files  not present ==========")
            print("=============== Begin label creation ===============")
            self.__create_labels()


################################################################################
# CHECK IF LABELLED FILE EXIST
################################################################################

    def __check_compiled_files(self):
        current_compiled_files = os.listdir(PROCESSED_DATA_FOLDER)
        return FINAL_FILE in current_compiled_files

################################################################################
# Create labels
################################################################################

    def __create_labels(self):
        ## Code for label creation
        pass

    