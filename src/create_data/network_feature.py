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
TRANSACTIONS_LABELLED_FILE = folder_location.TRANSACTIONS_LABELLED_FILE
MERGED_RELATIONSHIP_FILE = folder_location.MERGED_RELATIONSHIP_FILE

FEATURES_FOLDER = folder_location.FEATURES_DATA_FOLDER
NETWORK_RAW_FOLDERS = folder_location.PROFILE_DATA_FOLDERS
FINAL_FILE_1 = "network_time_ind_feature.csv"
FINAL_FILE_2 = "network_time_dep_feature.csv"
FINAL_FILE_3 = "zscore_feature.csv"

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
        print("=== Network time_independent_features file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_1}')
    else: # Create features and save
        print("=== Network time_independent_features file not found, begin creating  ===")

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

        df_to_return.to_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_1}', index=False) # there are only 5 columns

    return df_to_return

################################################################################
# Create helper functions for time independent features and network analysis 
################################################################################

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

# Reconstruct the adjacency list from the DataFrame.
def reconstruct_adj_list(df):
    # We group by the 'source' column, and for each source, we create a list of tuples (target, attr_dict).
    grouped = list(df.groupby("source"))
    adj_list = {}
    # Wrap the outer loop with tqdm to monitor progress.
    for source, group in tqdm(grouped, desc="Reconstructing adjacency list", total=len(grouped)):
        edges = []
        for _, row in group.iterrows():
            target = row["target"]
            # Convert row to dictionary and drop 'source' and 'target'
            attr = row.to_dict()
            attr.pop("source", None)
            attr.pop("target", None)
            edges.append((target, attr))
        adj_list[source] = edges
    return adj_list

# Define Lookup Functions to Precompute Memberships
def get_active_house(dt, house_by_date, house_dates):
    """
    Given a datetime dt, returns the set of active House member Littlesis IDs as of dt.
    Uses house_membership_by_date (sorted by date).
    """
    i = bisect.bisect_right(house_dates, dt) - 1
    return set() if i < 0 else set(house_by_date[house_dates[i]])

def get_imp_cands(dt, tic, tic_to_subcomm, subcomm_by_date, subcomm_dates):
    """
    Given a transaction date dt and a TIC, returns the union of active subcommittee member sets.
    For each subcommittee associated with the TIC (via tic_to_subcomm_mapper),
    it looks up the last change date (from house_date_subcomm_mapper) as of dt and
    accumulates the active member Littlesis IDs.
    """
    s = set()
    for sub in tic_to_subcomm.get(tic, ()):
        dates = subcomm_dates.get(sub, [])
        j = bisect.bisect_right(dates, dt) - 1
        if j >= 0:
            s |= set(subcomm_by_date[sub][dates[j]])
    return s

def bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, threshold:int = 3):
    '''
    Perform BFS and compute membership intersections for each source.
    Returns imp_map and full_map dictionaries.
    '''
    
    imp_map, full_map = {}, {}

    for source, grp in tqdm( 
        df_txns.groupby("id"), 
        desc="Compute by source", 
        total=df_txns["id"].nunique()
    ):
        if pd.isna(source):
            for idx in grp.index:
                imp_map[idx] = full_map[idx] = 0
            continue

        # Retrieve nodes reachable from the source within the given threshold.
        reachable = {str(n) for n in G.neighborhood(vertices=int(source), order=threshold, mode="out")}

        # Group transactions by the (TRANS_DATE, ISSUERTRADINGSYMBOL) pair.
        for (dt, tic), sub in grp.groupby(["TRANS_DATE", "ISSUERTRADINGSYMBOL"]):
            full_cands = full_by_date.get(dt, set())
            imp_cands = imp_by_dt_tic.get((dt, tic), set())
            full_cnt = len(full_cands & reachable)
            imp_cnt = len(imp_cands & reachable)
            for idx in sub.index:
                full_map[idx] = full_cnt
                imp_map[idx] = imp_cnt

        del reachable  # Free up memory before processing the next source.
    
    return imp_map, full_map


################################################################################
# Create time dependent features
################################################################################

def create_time_dependent_features():
    """This function will create the key-feature csv if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE_2 in current_compiled_files:
        print("=== Network time_dependent_features file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_2}')

    else: # Create features and save
        print("=== Network time_dependent_features file not found, begin creating  ===")

        #############################
        # create transaction to little sis node id match
        ##############################

        # merge nodeid with transaction data
        df_name_match = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/final_final_name_match.csv")
        mapping_dict = df_name_match.set_index("SEC_RPTOWNERCIK")["NODEID"].to_dict()
        df_txns = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/{TRANSACTIONS_LABELLED_FILE}",
                      usecols=["TRANS_SK", "ACCESSION_NUMBER", "TRANS_DATE", "RPTOWNERCIK_;", "ISSUERTRADINGSYMBOL"],
                      parse_dates=["TRANS_DATE"])
        df_txns["id"] = df_txns["RPTOWNERCIK_;"].map(mapping_dict)
        df_txns = df_txns.reset_index().rename(columns={"index": "orig_index"})
        df_txns.sort_values(["id","TRANS_DATE"], inplace=True)

        if df_txns.shape != (3171001, 7):
            print("Transaction Dataframe expected 3171001 rows 6 columns but has ", df_txns.shape)
        ## This resulting df_txns is saved as the file "txns_for_features.csv"

        #############################
        # load adjacency list to build graph
        ##############################
        edges_df = pd.read_csv(f"{NETWORK_RAW_FOLDERS}/adjacency_list.csv")
        
        adj_list_reconstructed = reconstruct_adj_list(edges_df)

        edges = []
        for src, nbrs in adj_list_reconstructed.items():
            s = int(src)
            for nbr, _ in nbrs:
                edges.append((s, int(nbr)))
        max_node = max((u for u, _ in edges), default=0)
        G = ig.Graph(n=max_node+1, edges=edges, directed=True)

        #############################
        # Create important_connections, full_congress_connections 
        ##############################

        # Ensure data exists
        network_data = os.listdir(NETWORK_RAW_FOLDERS)
        for file_required in ["house_membership_by_date.pkl", "tic_to_subcomm_mapper.pkl", "house_date_subcomm_mapper.pkl"]:
            if file_required not in network_data:
                print(f"--- {file_required} file not found. Please download from shared google drive: raw/profile_data/---")
                print("Alternatively, refer to ../notebook_references/create_network_files.ipynb ----")
            return FileNotFoundError(f"--- Aborting ---")

        # Load data 
        house_by_date = load_pickle(f"{NETWORK_RAW_FOLDERS}/house_membership_by_date.pkl")
        tic_to_subcomm = load_pickle(f"{NETWORK_RAW_FOLDERS}/tic_to_subcomm_mapper.pkl")
        subcomm_by_date = load_pickle(f"{NETWORK_RAW_FOLDERS}/house_date_subcomm_mapper.pkl")
            
        house_dates   = sorted(house_by_date.keys())
        subcomm_dates = {sub: sorted(tl.keys()) for sub, tl in subcomm_by_date.items()}
        
        # Precompute active House membership for each unique transaction date.
        unique_dates = df_txns["TRANS_DATE"].dropna().unique()
        full_by_date = {dt: get_active_house(dt, house_by_date, house_dates) for dt in tqdm(unique_dates, desc="Precompute full-house")}

        # Precompute important candidate memberships (from subcommittees) for each unique (TRANS_DATE, ISSUERTRADINGSYMBOL) pair.
        unique_dt_tic = df_txns[["TRANS_DATE", "ISSUERTRADINGSYMBOL"]].drop_duplicates().values
        imp_by_dt_tic = {
            (pd.Timestamp(dt), tic): get_imp_cands(pd.Timestamp(dt), tic, tic_to_subcomm, subcomm_by_date, subcomm_dates)
            for dt, tic in tqdm(unique_dt_tic, desc="Precompute imp cands")
        }

        # Compute BFS and membership intersections for each source.
        ## BFS Threshold = 3
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 3)
        df_txns["important_connections"]     = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        ## BFS Threshold = 2
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 2)
        df_txns["house_t2_important_connections"]     = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["house_t2_full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        ## BFS Threshold = 1
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 1)
        df_txns["house_t1_important_connections"]     = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["house_t1_full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        
        #############################
        # sen_important_connections, sen_full_congress_connections 
        ##############################

        # Ensure data exists
        network_data = os.listdir(NETWORK_RAW_FOLDERS)
        for file_required in ["senate_membership_by_date.pkl", "sen_tic_to_subcomm_mapper.pkl", "senate_date_subcomm_mapper.pkl"]:
            if file_required not in network_data:
                print(f"--- {file_required} file not found. Please download from shared google drive: raw/profile_data/---")
                print("Alternatively, refer to ../notebook_references/create_network_files.ipynb ----")
            return FileNotFoundError(f"--- Aborting ---")
        
        house_by_date = load_pickle(f"{NETWORK_RAW_FOLDERS}/senate_membership_by_date.pkl")
        tic_to_subcomm = load_pickle(f"{NETWORK_RAW_FOLDERS}/sen_tic_to_subcomm_mapper.pkl")
        subcomm_by_date = load_pickle(f"{NETWORK_RAW_FOLDERS}/senate_date_subcomm_mapper.pkl")
        
        # Create sorted lists for quick lookup.
        house_dates = sorted(house_by_date.keys())
        # For each subcommittee, get sorted change dates.
        subcomm_dates = {sub: sorted(tl.keys()) for sub, tl in subcomm_by_date.items()}

        # Precompute active House membership for each unique transaction date.
        unique_dates = df_txns["TRANS_DATE"].dropna().unique()
        full_by_date = {dt: get_active_house(dt, house_by_date, house_dates) for dt in tqdm(unique_dates, desc="Precompute full-house")}

        # Precompute important candidate memberships (from subcommittees) for each unique (TRANS_DATE, ISSUERTRADINGSYMBOL) pair.
        unique_dt_tic = df_txns[["TRANS_DATE", "ISSUERTRADINGSYMBOL"]].drop_duplicates().values
        imp_by_dt_tic = {
            (pd.Timestamp(dt), tic): get_imp_cands(pd.Timestamp(dt), tic, tic_to_subcomm, subcomm_by_date, subcomm_dates)
            for dt, tic in tqdm(unique_dt_tic, desc="Precompute imp cands")
        }

        ## BFS Threshold = 3
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 3)
        df_txns["sen_important_connections"] = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["sen_full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        ## BFS Threshold = 2
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 2)
        df_txns["sen_t2_important_connections"] = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["sen_t2_full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        ## BFS Threshold = 2
        imp_map, full_map = bfs_and_compute_membership_intersection(df_txns, G, full_by_date, imp_by_dt_tic, 1)
        df_txns["sen_t1_important_connections"] = df_txns["orig_index"].map(imp_map).fillna(0).astype(int)
        df_txns["sen_t1_full_congress_connections"] = df_txns["orig_index"].map(full_map).fillna(0).astype(int)

        print("file created with columns: " , df_txns.columns, "and shape: ", df_txns.shape)

        df_txns.to_csv(f"{FEATURES_FOLDER}/{FINAL_FILE_2}", index=False)
        df_to_return = df_txns

    return df_to_return

################################################################################
# Create zscore features
################################################################################

def create_zscore_features():
    """This function will create the key-feature csv if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE_3 in current_compiled_files:
        print("=== Network zscore_features file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE_3}')

    else: # Create features and save
        print("=== Network zscore_features file not found, begin creating  ===")
        print("--- Please refer to notebook --- ")
        df_to_return = pd.DataFrame()
        
    return df_to_return