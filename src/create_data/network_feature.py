import pandas as pd
import numpy as np
import os
import sys
import glob
import pickle
from datetime import datetime
from collections import deque
from tqdm.notebook import tqdm
import re
import bisect
import igraph as ig

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER
ABNORMAL_CSV = folder_location.ABNORMAL_CSV
MERGED_RELATIONSHIP_FILE = folder_location.MERGED_RELATIONSHIP_FILE

FEATURES_FOLDER = folder_location.FEATURES_DATA_FOLDER
NETWORK_RAW_FOLDERS = folder_location.PROFILE_DATA_FOLDERS
FINAL_FILE_1 = "network_time_ind_feature.csv"
FINAL_FILE_2 = "network_time_dep_feature.csv"

################################################################################
# Create time independent features (matches on RPTOWNERCIK_;)
################################################################################

def create_time_independent_features():
    """This function will create the key-feature csv if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE_1 in current_compiled_files:
        print("=== Network Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_1}')
    else: # Create features and save
        print("=== Network Key file not found, begin creating  ===")

        # load relationship data
        network_df = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/merged_relationships_full.csv') # (1817762, 146)
        if network_df.shape[0] != 1817762:
            print("Network Dataframe expected 1817762 rows but has ", network_df.shape[0])
        else:
            print("Network Dataframe loaded with expected shape: ", network_df.shape)
        # load node data
        entities = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/entities_merged.csv') # (431898, 65)
        if entities.shape[0] != 431898:
            print("Entities Dataframe expected 431898 rows but has ", entities.shape[0])

        ##############################
        # is_lobby, has_lobby
        ##############################

        # is_lobby: based on node type, boolean
        nodes_lobby = entities[entities['types'].str.lower().str.contains('lobby').fillna(False)][['id']].drop_duplicates()
        assert nodes_lobby.shape[0] == nodes_lobby['id'].nunique() # each row is a unique entity id
        nodes_lobby = nodes_lobby.rename(columns={'id': 'NODEID'})
        nodes_lobby['is_lobby'] = True
    
        if nodes_lobby.shape[0] != 14507:
            print("is_lobby Dataframe expected 14507 rows but has ", nodes_lobby.shape[0]) 

        # has_lobby: based on realtionship of lobbying, category_id = 7
        give_lobby_nodes = network_df[(network_df['category_id']==7)][['entity1_id']].drop_duplicates()
        receive_lobby_nodes = network_df[(network_df['category_id']==7)][['entity2_id']].drop_duplicates()
        give_lobby_nodes = give_lobby_nodes.rename(columns={'entity1_id': 'NODEID'})
        receive_lobby_nodes = receive_lobby_nodes.rename(columns={'entity2_id': 'NODEID'})
        if give_lobby_nodes.shape[0] != 1741 or receive_lobby_nodes.shape[0] != 371:
            print("give_lobby_nodes Dataframe expected 1741 rows but has ", give_lobby_nodes.shape[0])
            print("receive_lobby_nodes Dataframe expected 371 rows but has ", receive_lobby_nodes.shape[0])

        ## Merge the two dataframes to get a single dataframe with all nodes that have existed in a lobby relationship
        lobby_rs = pd.merge(give_lobby_nodes, receive_lobby_nodes, how='outer', on='NODEID', indicator=True)
        lobby_rs['has_lobby'] = lobby_rs['_merge'].replace({'left_only': 'give', 'right_only': 'receive', 'both': 'give_and_receive'})
        lobby_rs.drop(columns=['_merge'], inplace=True)
        assert lobby_rs.shape[0] == lobby_rs['NODEID'].nunique() # each row is a unique entity id
        if lobby_rs.shape[0] != 2084:
            print("has_lobby Dataframe expected 2084 rows but has ", lobby_rs.shape[0])

        #############################
        # has_donate
        ##############################

        # has_donate: based on realtionship of donation, category_id = 5
        give_money_nodes = network_df[(network_df['category_id']==5)][['entity1_id']].drop_duplicates()
        receive_money_nodes = network_df[(network_df['category_id']==5)][['entity2_id']].drop_duplicates()
        give_money_nodes = give_money_nodes.rename(columns={'entity1_id': 'NODEID'})
        receive_money_nodes = receive_money_nodes.rename(columns={'entity2_id': 'NODEID'})
        if give_money_nodes.shape[0] != 92307 or receive_money_nodes.shape[0] != 31385:
            print("give_money_nodes Dataframe expected 92307 rows but has ", give_money_nodes.shape[0])
            print("receive_money_nodes Dataframe expected 31385 rows but has ", receive_money_nodes.shape[0])
        
        ## Merge the two dataframes to get a single dataframe with all nodes that have existed in a lobby relationship
        donation_rs = pd.merge(give_money_nodes, receive_money_nodes, how='outer', on='NODEID', indicator=True)
        donation_rs['has_donate'] = donation_rs['_merge'].replace({'left_only': 'give', 'right_only': 'receive', 'both': 'give_and_receive'})
        donation_rs.drop(columns=['_merge'], inplace=True)
        assert donation_rs.shape[0] == donation_rs['NODEID'].nunique() # each row is a unique entity id
        if donation_rs.shape[0] != 118572:
            print("has_donate Dataframe expected 2084 rows but has ", donation_rs.shape[0])
  
        ##############################
        # Merge all features
        ##############################
        # read name_match data
        name_match = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/final_final_name_match.csv')
        name_match = name_match.rename(columns={'SEC_RPTOWNERCIK': 'RPTOWNERCIK_;'})

        merge1 = name_match.merge(nodes_lobby, how='left', on='NODEID')
        merge2 = merge1.merge(lobby_rs, how='left', on='NODEID')
        assert merge2.shape[0] == merge1.shape[0] # to ensure no additional rows created
        df_to_return = merge2.merge(donation_rs, how='left', on='NODEID')
        assert df_to_return.shape[0] == merge2.shape[0] # each row is a unique entity id

        df_to_return.to_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_1}') # there are only 5 columns

    return df_to_return

################################################################################
# Create helper functions for time independent features and network analysis 
################################################################################

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

################################################################################
# Create time independent features
################################################################################

def create_time_independent_features():
    """This function will create the key-feature csv if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE_1 in current_compiled_files:
        print("=== Network Key 2 file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_2}')
    else: # Create features and save
        print("=== Network Key file not found, begin creating  ===")

        #############################
        # create transaction to node id match
        ##############################

        df_name_match = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/final_final_name_match.csv")
        mapping_dict = df_name_match.set_index("SEC_RPTOWNERCIK")["NODEID"].to_dict()
        df_txns = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/{ABNORMAL_CSV}",
                      usecols=["TRANS_SK", "ACCESSION_NUMBER", "TRANS_DATE", "RPTOWNERCIK", "ISSUERTRADINGSYMBOL"],
                      parse_dates=["TRANS_DATE"])
        df_txns["id"] = df_txns["RPTOWNERCIK"].map(mapping_dict)
        if df_txns.shape != (3171001, 6):
            print("Transaction Dataframe expected 3171001 rows 6 columns but has ", df_txns.shape)
        ## Caitlyn saves this to df_txns.to_csv("txns_for_features.csv", index=False)

        #############################
        # committee connections
        ##############################

        network_data = os.listdir(FEATURES_FOLDER)
        if "congress_nodeid_mapper.pkl" in network_data: 
            congress_nodeid_mapper = load_pickle(f"{NETWORK_RAW_FOLDERS}/congress_nodeid_mapper.pkl")
        else:
            # create congress_nodeid_mapper
            pass
        
        if "congress_date_subcomm_mapper.pkl" in network_data:
            congress_date_subcomm_mapper = load_pickle(f"{NETWORK_RAW_FOLDERS}/congress_date_subcomm_mapper.pkl")
        else:
            # create congress_date_subcomm_mapper
            pass

        if "tic_to_subcomm_mapper.pkl" in network_data:
            tic_to_subcomm_mapper = load_pickle(f"{NETWORK_RAW_FOLDERS}/tic_to_subcomm_mapper.pkl")
        else:
            pass
        
        #############################
        # committee connections
        ##############################
    
    pass