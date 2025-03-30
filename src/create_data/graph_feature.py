import pandas as pd
import numpy as np
import os

PROCESSED_DATA_FOLDER = "../data_untracked/processed"
ABNORMAL_CSV = "snorkel_labels.csv"

FINAL_FOLDER = "../data_untracked/features"
FINAL_FILE = "graph_feature.csv"

def create_features():
    """This function will create the key-feature graph if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FINAL_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Graph Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FINAL_FOLDER}/{FINAL_FILE}', parse_dates=['TRANS_DATE'])
    else: # Create features and save
        print("=== Graph Key file not found, begin creating  ===")
        
        ## Merging
        # Define columns to drop for entity1 and entity2
        cols_to_drop = [
            # entity1 columns
            "entity1_ext_Business_annual_profit",
            "entity1_ext_Business_assets",
            "entity1_ext_Business_aum",
            "entity1_ext_Business_marketcap",
            "entity1_ext_Business_net_income",
            "entity1_ext_ElectedRepresentative_bioguide_id",
            "entity1_ext_ElectedRepresentative_crp_id",
            "entity1_ext_ElectedRepresentative_govtrack_id",
            "entity1_ext_ElectedRepresentative_pvs_id",
            "entity1_ext_ElectedRepresentative_watchdog_id",
            "entity1_ext_GovernmentBody_city",
            "entity1_ext_GovernmentBody_county",
            "entity1_ext_GovernmentBody_state_id",
            "entity1_ext_Lobbyist_lda_registrant_id",
            "entity1_ext_Org_employees",
            "entity1_ext_Org_fedspending_id",
            "entity1_ext_Org_lda_registrant_id",
            "entity1_ext_Org_name",
            "entity1_ext_Org_name_nick",
            "entity1_ext_Org_revenue",
            "entity1_ext_Person_birthplace",
            "entity1_ext_Person_gender_id",
            "entity1_ext_Person_is_independent",
            "entity1_ext_Person_name_first",
            "entity1_ext_Person_name_last",
            "entity1_ext_Person_name_maiden",
            "entity1_ext_Person_name_middle",
            "entity1_ext_Person_name_nick",
            "entity1_ext_Person_name_prefix",
            "entity1_ext_Person_name_suffix",
            "entity1_ext_Person_nationality",
            "entity1_ext_Person_net_worth",
            "entity1_ext_Person_party_id",
            "entity1_ext_PoliticalCandidate_crp_id",
            #"entity1_ext_PoliticalCandidate_house_fec_id",
            "entity1_ext_PoliticalCandidate_is_federal",
            "entity1_ext_PoliticalCandidate_is_local",
            "entity1_ext_PoliticalCandidate_is_state",
            "entity1_ext_PoliticalCandidate_pres_fec_id",
            #"entity1_ext_PoliticalCandidate_senate_fec_id",
            "entity1_ext_PoliticalFundraising_fec_id",
            "entity1_ext_PoliticalFundraising_state_id",
            "entity1_ext_PoliticalFundraising_type_id",
            "entity1_ext_School_endowment",
            "entity1_ext_School_faculty",
            "entity1_ext_School_is_private",
            "entity1_ext_School_students",
            "entity1_ext_School_tuition",
            #"entity1_link_self",
            "entity1_updated_at",
            "entity1_website",
            
            # entity2 columns
            "entity2_ext_Business_annual_profit",
            "entity2_ext_Business_assets",
            "entity2_ext_Business_aum",
            "entity2_ext_Business_marketcap",
            "entity2_ext_Business_net_income",
            "entity2_ext_ElectedRepresentative_bioguide_id",
            "entity2_ext_ElectedRepresentative_crp_id",
            "entity2_ext_ElectedRepresentative_govtrack_id",
            "entity2_ext_ElectedRepresentative_pvs_id",
            "entity2_ext_ElectedRepresentative_watchdog_id",
            "entity2_ext_GovernmentBody_city",
            "entity2_ext_GovernmentBody_county",
            "entity2_ext_GovernmentBody_state_id",
            "entity2_ext_Lobbyist_lda_registrant_id",
            "entity2_ext_Org_employees",
            "entity2_ext_Org_fedspending_id",
            "entity2_ext_Org_lda_registrant_id",
            "entity2_ext_Org_name",
            "entity2_ext_Org_name_nick",
            "entity2_ext_Org_revenue",
            "entity2_ext_Person_birthplace",
            "entity2_ext_Person_gender_id",
            "entity2_ext_Person_is_independent",
            "entity2_ext_Person_name_first",
            "entity2_ext_Person_name_last",
            "entity2_ext_Person_name_maiden",
            "entity2_ext_Person_name_middle",
            "entity2_ext_Person_name_nick",
            "entity2_ext_Person_name_prefix",
            "entity2_ext_Person_name_suffix",
            "entity2_ext_Person_nationality",
            "entity2_ext_Person_net_worth",
            "entity2_ext_Person_party_id",
            "entity2_ext_PoliticalCandidate_crp_id",
            #"entity2_ext_PoliticalCandidate_house_fec_id",
            "entity2_ext_PoliticalCandidate_is_federal",
            "entity2_ext_PoliticalCandidate_is_local",
            "entity2_ext_PoliticalCandidate_is_state",
            "entity2_ext_PoliticalCandidate_pres_fec_id",
            #"entity2_ext_PoliticalCandidate_senate_fec_id",
            "entity2_ext_PoliticalFundraising_fec_id",
            "entity2_ext_PoliticalFundraising_state_id",
            "entity2_ext_PoliticalFundraising_type_id",
            "entity2_ext_School_endowment",
            "entity2_ext_School_faculty",
            "entity2_ext_School_is_private",
            "entity2_ext_School_students",
            "entity2_ext_School_tuition",
            #"entity2_link_self",
            "entity2_updated_at",
            "entity2_website"
        ]

        # Load the CSV file
        df_relationships = pd.read_csv(f"{PROCESSED_DATA_FOLDER}/merged_relationships_full.csv")

        # Drop the specified columns
        df_relationships = df_relationships.drop(columns=cols_to_drop, errors='ignore')

        #lowercase for merging
        df_relationships['entity1_name'] = df_relationships['entity1_name'].str.lower().str.strip()

        matches = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/name_match_improved.csv')

        threshold = 7.5 # for matching score 
        matches = matches[matches['score'] >= threshold]

        #lower for easier merging
        matches['littlesis'] = matches['littlesis'].str.lower().str.strip()
        matches['form4'] = matches['form4'].str.lower().str.strip()

        merged_entity = pd.merge(
            matches,
            df_relationships,
            right_on='entity1_name',
            left_on='littlesis',
            how='inner'
        )

        # Converting nan to False 
        merged_entity['cat_is_executive'] = merged_entity['cat_is_executive'].fillna(False)
        merged_entity['cat_is_board'] = merged_entity['cat_is_board'].fillna(False)
        merged_entity['is_current'] = merged_entity['is_current'].fillna(False)

        df_form4 = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{ABNORMAL_CSV}')

        # Only looking at 1 RPTOWNER for now
        df_form4 = df_form4[df_form4['NUM_RPTOWNERCIK_;'] == 1]

        # Include columns as needed 
        columns_need = ['ACCESSION_NUMBER', 'TRANS_SK', 'TRANS_DATE', 'RPTOWNERNAME_;']
        
        # These were used in EDA codes
        '''['CAR_5_before', 'CAR_5_after', 'CAR_30_before', 'CAR_30_after', 'CAR_60_before',
        'CAR_60_after', 'CAR_120_before', 'CAR_120_after',
        'effective_CAR_30_after', 'effective_CAR_60_after','effective_CAR_120_after', 
        'local_score_30', 'n_local_30','isolation_raw_30','local_mean_30', 'local_std_30', 'n_local_unq_30',
        'local_score_60', 'n_local_60', 'isolation_raw_60','local_mean_60','local_std_60', 'n_local_unq_60',
        'local_score_120', 'n_local_120', 'isolation_raw_120','local_mean_120', 'local_std_120','n_local_unq_120',
        'isolation_z_30','isolation_z_60', 'isolation_z_120', 'anomaly_score_30',
        'anomaly_score_60', 'anomaly_score_120',
        'local_score_30_sig', 'isolation_z_30_sig',
        'anomaly_score_30_sig', 'local_score_60_sig', 'isolation_z_60_sig',
        'anomaly_score_60_sig', 'local_score_120_sig', 'isolation_z_120_sig',
        'anomaly_score_120_sig']'''

        #Update columns for merging 
        df_form4['RPTOWNERNAME_;'] = df_form4['RPTOWNERNAME_;'].str.lower().str.strip()
        df_form4['TRANS_DATE'] = pd.to_datetime(df_form4['TRANS_DATE'], errors='coerce')

        df_form4 = df_form4[columns_need].copy()

        merged_for_features = pd.merge(
            df_form4,
            merged_entity,
            left_on='RPTOWNERNAME_;',
            right_on='form4',
            how='inner'  #we only care about those with relations for now
        )

        merged_for_features = merged_for_features.drop_duplicates()

        ##############################
        # Lobbyist Score Feature
        ##############################

        # Filter rows where 'entity2_types' contains "lobbyist" or "lobbying" (case insensitive)
        lobbyist_mask = merged_for_features['entity2_types'].str.contains('lobbyist|lobbying', case=False, na=False)

        # Get the unique values that match the condition
        unique_lobbyist_values = merged_for_features.loc[lobbyist_mask, 'entity2_types'].unique()

        # Updated function to compute the lobbyist score per edge,
        # incorporating transaction date filtering.
        def edge_lobbyist_score(row):
            score = 0
            # Check if entity2_types indicates a lobbying-related connection
            if pd.notnull(row.get('entity2_types')):
                et2 = row['entity2_types'].lower()
                if "lobbying" in et2 or "lobbyist" in et2:
                    # Get transaction date, start date, and end date (assumed to be datetime columns)
                    trans_date = row.get('TRANS_DATE_dt', None)
                    start = row.get('start_date_dt', None)
                    end = row.get('end_date_dt', None)
                    # If both relationship dates exist, count only if transaction date falls between them.
                    if pd.notnull(start) and pd.notnull(end):
                        if pd.notnull(trans_date) and (start <= trans_date <= end):
                            score += 1
                    else:
                        # If either start or end date is missing, count the edge as valid.
                        score += 1
            return score
        
        merged_for_features['lobbyist_score'] = merged_for_features.apply(edge_lobbyist_score, axis=1)
        agg_features_by_entity = merged_for_features.groupby('entity1_id')['lobbyist_score'].sum().reset_index()
        agg_features_by_entity.rename(columns={'lobbyist_score': 'total_edge_lobbyist_score'}, inplace=True)

        # Compute a one-time bonus for each entity based on whether the entity (from entity1_types) is itself a lobbyist.
        def entity_is_lobbyist(types_str):
            if pd.isnull(types_str):
                return 0
            return 1 if ("lobbying" in types_str.lower() or "lobbyist" in types_str.lower()) else 0

        # For each reporting owner (entity1_id), take a representative value of entity1_types and compute the bonus.
        bonus_by_entity = merged_for_features.groupby('entity1_id')['entity1_types'].first().reset_index()
        bonus_by_entity['lobbyist_bonus'] = bonus_by_entity['entity1_types'].apply(entity_is_lobbyist)
        bonus_by_entity = bonus_by_entity[['entity1_id', 'lobbyist_bonus']]

        # Merge the aggregated edge scores with the bonus.
        agg_by_entity = pd.merge(agg_features_by_entity, bonus_by_entity, on='entity1_id', how='outer')
        agg_by_entity['total_edge_lobbyist_score'] = agg_by_entity['total_edge_lobbyist_score'].fillna(0)
        agg_by_entity['lobbyist_bonus'] = agg_by_entity['lobbyist_bonus'].fillna(0)
        agg_by_entity['lobbyist_score_final'] = agg_by_entity['total_edge_lobbyist_score'] + agg_by_entity['lobbyist_bonus']

        rep_transactions = merged_for_features.groupby('TRANS_SK').first().reset_index()

        # Merge Aggregated Entity-Level Features into the Transaction-Level Table
        # Merge on entity1_id so that each transaction gets the same aggregated lobbyist score for its reporting owner.
        final_summary = pd.merge(rep_transactions,
                                agg_by_entity[['entity1_id', 'total_edge_lobbyist_score', 'lobbyist_bonus', 'lobbyist_score_final']],
                                on='entity1_id', how='left')
        
        ##############################
        # No. of connections to Senators Feature
        ##############################

        # Select only relevant columns and drop duplicate relationships
        unique_edges = merged_for_features[[
            "entity1_id",
            "entity2_id",
            "entity1_ext_PoliticalCandidate_senate_fec_id",
            "entity2_ext_PoliticalCandidate_senate_fec_id"
        ]].drop_duplicates()

        # Create the senator connection counter
        num_connections_to_senators = {}

        # Loop through the unique edges
        for _, row in unique_edges.iterrows():
            entity1_id = row["entity1_id"]
            entity2_id = row["entity2_id"]
            
            # Check senator status
            is_entity1_senator = pd.notna(row.get("entity1_ext_PoliticalCandidate_senate_fec_id"))
            is_entity2_senator = pd.notna(row.get("entity2_ext_PoliticalCandidate_senate_fec_id"))
            
            # Count one connection per senator relationship (undirected)
            if is_entity1_senator:
                num_connections_to_senators[entity2_id] = num_connections_to_senators.get(entity2_id, 0) + 1
            if is_entity2_senator:
                num_connections_to_senators[entity1_id] = num_connections_to_senators.get(entity1_id, 0) + 1

        # Convert to DataFrame
        senator_conn_df = pd.DataFrame(list(num_connections_to_senators.items()), columns=["entity_id", "num_connections_to_senators"])

        # Create two separate senator connection DataFrames
        # One for entity1_id (person/insider)
        senator_conn_entity1 = senator_conn_df.rename(columns={
            "entity_id": "entity1_id",
            "num_connections_to_senators": "entity1_num_connections_to_senators"
        })

        # One for entity2_id (company/org)
        senator_conn_entity2 = senator_conn_df.rename(columns={
            "entity_id": "entity2_id",
            "num_connections_to_senators": "entity2_num_connections_to_senators"
        })

        # Merge both into final_summary
        final_summary = final_summary.merge(senator_conn_entity1, on="entity1_id", how="left")
        final_summary = final_summary.merge(senator_conn_entity2, on="entity2_id", how="left")

        # Fill missing values with 0 (no connections)
        final_summary["entity1_num_connections_to_senators"] = final_summary["entity1_num_connections_to_senators"].fillna(0).astype(int)
        final_summary["entity2_num_connections_to_senators"] = final_summary["entity2_num_connections_to_senators"].fillna(0).astype(int)

        ##############################
        # No. of connections to House Feature
        ##############################
        # Select only relevant columns and drop duplicate relationships
        unique_edges = merged_for_features[[
            "entity1_id",
            "entity2_id",
            "entity1_ext_PoliticalCandidate_house_fec_id",
            "entity2_ext_PoliticalCandidate_house_fec_id"
        ]].drop_duplicates()

        # Create the house connection counter
        num_connections_to_house = {}

        # Loop through the unique edges
        for _, row in unique_edges.iterrows():
            entity1_id = row["entity1_id"]
            entity2_id = row["entity2_id"]
            
            # Check house status
            is_entity1_house = pd.notna(row.get("entity1_ext_PoliticalCandidate_house_fec_id"))
            is_entity2_house = pd.notna(row.get("entity2_ext_PoliticalCandidate_house_fec_id"))
            
            # Count one connection per house relationship (undirected)
            if is_entity1_house:
                num_connections_to_house[entity2_id] = num_connections_to_house.get(entity2_id, 0) + 1
            if is_entity2_house:
                num_connections_to_house[entity1_id] = num_connections_to_house.get(entity1_id, 0) + 1

        # Convert to DataFrame
        house_conn_df = pd.DataFrame(list(num_connections_to_house.items()), columns=["entity_id", "num_connections_to_house"])

        # Create two separate house connection DataFrames
        # One for entity1_id (person/insider)
        house_conn_entity1 = house_conn_df.rename(columns={
            "entity_id": "entity1_id",
            "num_connections_to_house": "entity1_num_connections_to_house"
        })

        # One for entity2_id (company/org)
        house_conn_entity2 = house_conn_df.rename(columns={
            "entity_id": "entity2_id",
            "num_connections_to_house": "entity2_num_connections_to_house"
        })

        # Merge both into final_summary
        final_summary = final_summary.merge(house_conn_entity1, on="entity1_id", how="left")
        final_summary = final_summary.merge(house_conn_entity2, on="entity2_id", how="left")

        # Fill missing values with 0 (no connections)
        final_summary["entity1_num_connections_to_house"] = final_summary["entity1_num_connections_to_house"].fillna(0).astype(int)
        final_summary["entity2_num_connections_to_house"] = final_summary["entity2_num_connections_to_house"].fillna(0).astype(int)

        final_summary["total_senate_connections"] = (
            final_summary["entity1_num_connections_to_senators"] +
            final_summary["entity2_num_connections_to_senators"]
        )

        final_summary["total_house_connections"] = (
            final_summary["entity1_num_connections_to_house"] +
            final_summary["entity2_num_connections_to_house"]
        )

        ##############################
        # Combined Seniority Score Feature
        ##############################
        from collections import Counter

        # List of possible seniority keywords (no scores yet)
        seniority_keywords_raw = [
            "lead independent director", "chairperson", "chief executive", "coo", "treasurer",
            "cfo", "managing director", "secretary", "vp", "officer", "chairman", "ceo",
            "executive", "vice president", "board", "president", "director"
        ]

        # Combine and lowercase descriptions
        merged_for_features["combined_desc"] = (
            merged_for_features["description1"].fillna("") + " " +
            merged_for_features["description2"].fillna("")
        ).str.lower()

        # Count frequencies
        keyword_counts = Counter()
        for desc in merged_for_features["combined_desc"]:
            for keyword in seniority_keywords_raw:
                if keyword in desc:
                    keyword_counts[keyword] += 1

        # Create DataFrame with frequency info
        keyword_freq_df = pd.DataFrame.from_dict(keyword_counts, orient="index", columns=["count"])
        keyword_freq_df["frequency_percent"] = 100 * keyword_freq_df["count"] / len(df_relationships)
        keyword_freq_df = keyword_freq_df.sort_values("frequency_percent", ascending=True)

        # Final seniority score mapping from earlier
        seniority_keywords = {
            "lead independent director": 3,
            "chairperson": 3,
            "chief executive": 3,
            "coo": 3,
            "treasurer": 2,
            "cfo": 2,
            "managing director": 2,
            "secretary": 2,
            "vp": 2,
            "officer": 2,
            "chairman": 2,
            "ceo": 2,
            "executive": 1,
            "vice president": 1,
            "board": 1,
            "president": 1,
            "director": 1
        }

        def get_empirical_seniority(row):
            score = 0
            desc = f"{str(row['description1'])} {str(row['description2'])}".lower()
            
            for keyword, val in seniority_keywords.items():
                if keyword in desc:
                    score = max(score, val)
            
            # Fallbacks
            if row.get("cat_is_board", 0) == 1:
                score = max(score, 2)
            if row.get("cat_is_executive", 0) == 1:
                score = max(score, 1)

            return score

        # Apply the function
        df_relationships["position_seniority_score"] = df_relationships.apply(get_empirical_seniority, axis=1)

        # Most senior position each person held
        seniority_scores_entity1 = df_relationships.groupby("entity1_id")["position_seniority_score"].max().reset_index()
        seniority_scores_entity1.rename(columns={"position_seniority_score": "entity1_position_seniority_index"}, inplace=True)

        seniority_scores_entity2 = df_relationships.groupby("entity2_id")["position_seniority_score"].max().reset_index()
        seniority_scores_entity2.rename(columns={"position_seniority_score": "entity2_position_seniority_index"}, inplace=True)

        final_summary = final_summary.merge(seniority_scores_entity1, on="entity1_id", how="left")
        final_summary = final_summary.merge(seniority_scores_entity2, on="entity2_id", how="left")

        final_summary["combined_seniority_score"] = (
            final_summary["entity1_position_seniority_index"].fillna(0) +
            final_summary["entity2_position_seniority_index"].fillna(0)
        )

        ##############################
        # PI Combined Feature
        ##############################

        # Define political keywords
        political_keywords = {"elected representative", "government body"}

        # Initialize counters
        individual_pi = {}
        firm_pi = {}

        # Loop through relationships
        for _, row in df_relationships.iterrows():
            entity1_id = row["entity1_id"]
            entity2_id = row["entity2_id"]
            
            entity1_types = str(row.get("entity1_types", "")).lower()
            entity2_types = str(row.get("entity2_types", "")).lower()
            
            # If entity2 is political, entity1 is connected to politics
            if any(keyword in entity2_types for keyword in political_keywords):
                individual_pi[entity1_id] = individual_pi.get(entity1_id, 0) + 1
                firm_pi[entity1_id] = firm_pi.get(entity1_id, 0) + 0  # only count individual here

            # If entity1 is political, entity2 is connected
            if any(keyword in entity1_types for keyword in political_keywords):
                firm_pi[entity2_id] = firm_pi.get(entity2_id, 0) + 1
                individual_pi[entity2_id] = individual_pi.get(entity2_id, 0) + 0  # only count firm here

        # Convert to DataFrames
        df_individual_pi = pd.DataFrame.from_dict(individual_pi, orient='index').reset_index()
        df_individual_pi.columns = ["entity_id", "PI_individual"]

        df_firm_pi = pd.DataFrame.from_dict(firm_pi, orient='index').reset_index()
        df_firm_pi.columns = ["entity_id", "PI_firm"]

        # Merge the two together
        pi_combined_df = pd.merge(df_individual_pi, df_firm_pi, on="entity_id", how="outer").fillna(0)

        # Compute combined PI
        pi_combined_df["PI_combined"] = pi_combined_df["PI_individual"] + pi_combined_df["PI_firm"]

        # Rename columns for entity1 (insider)
        pi_entity1 = pi_combined_df.rename(columns={
            "entity_id": "entity1_id",
            "PI_individual": "entity1_PI_individual",
            "PI_firm": "entity1_PI_firm",
            "PI_combined": "PI_combined_entity1"
        })

        # Rename columns for entity2 (firm/org)
        pi_entity2 = pi_combined_df.rename(columns={
            "entity_id": "entity2_id",
            "PI_individual": "entity2_PI_individual",
            "PI_firm": "entity2_PI_firm",
            "PI_combined": "PI_combined_entity2"
        })

        # Merge into final_summary
        final_summary = final_summary.merge(pi_entity1, on="entity1_id", how="left")
        final_summary = final_summary.merge(pi_entity2, on="entity2_id", how="left")

        # Fill missing values with 0
        final_summary[[
            "PI_combined_entity1", "PI_combined_entity2",
            "entity1_PI_individual", "entity1_PI_firm",
            "entity2_PI_individual", "entity2_PI_firm"
        ]] = final_summary[[
            "PI_combined_entity1", "PI_combined_entity2",
            "entity1_PI_individual", "entity1_PI_firm",
            "entity2_PI_individual", "entity2_PI_firm"
        ]].fillna(0).astype(int)

        # Add a total PI_combined across both
        final_summary["PI_combined_total"] = (
            final_summary["PI_combined_entity1"] + final_summary["PI_combined_entity2"]
        )

        ##############################
        # Save file
        ##############################
        #features_to_keep = ["lobbyist_score_final", "num_connections_to_senators", "num_connections_to_house", "position_seniority_score", "PI_combined"]
        features_to_keep = ["lobbyist_score_final", "total_senate_connections", "total_house_connections", "combined_seniority_score", "PI_combined_total"]
        key = ["ACCESSION_NUMBER", "TRANS_SK", "TRANS_DATE", "RPTOWNERNAME_;"]

        df_to_save = final_summary[features_to_keep + key]
        df_to_save.to_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save

    return df_to_return