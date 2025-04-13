import os

#####################
# Processed folder
#####################
PROCESSED_DATA_FOLDER = '../../data_untracked/processed'
TRANSACTIONS_MERGED_FILE = 'transactions_final.csv'
TRANSACTIONS_ABNORMAL_FILE = 'transactions_abnormal_returns.csv'
TRANSACTIONS_ABNORMAL_ANOMALY_FILE = 'transactions_abnormal_returns_anomaly_score.csv'
ABNORMAL_CSV = 'merged_txns_SNORKEL.csv' # renamed from ABNORMAL_CSV
FULL_FEATURES_FILE = 'full_features.csv'
TRAINING_FULL_FEATURES_FILE = 'training_full_features.csv'
TESTING_FULL_FEATURES_FILE = 'testing_full_features.csv'

#####################
# Features folder
#####################
FEATURES_DATA_FOLDER = '../../data_untracked/features'
MERGED_RELATIONSHIP_FILE = 'merged_relationships_full.csv'

#####################
# SEC folder
#####################
SEC_SUBMISSIONS_FOLDER = '../../data_untracked/raw/sec_submissions'
SEC_DATA_FOLDER = "../../data_untracked/raw/sec_submissions/compiled"
FOOTNOTE_FILE = "FOOTNOTES.csv"

#####################
# Profile folder
#####################
PROFILE_DATA_FOLDERS = '../../data_untracked/raw/profile_data'

#####################
# Abnormal returns folder
#####################
ABNORMAL_RETURNS_FOLDER = '../../data_untracked/raw/abnormal_returns'
STOCK_DATA_FOLDER = 'crsp/daily_stock_data_by_ticker'
DAILY_BETA_FOLDER = 'crsp/split_csv_parts'
RISK_FREE_RATE_DATA = 'fred/risk_free_rate_daily.csv'
TRANS_ARS_FOLDER = 'trans_ars'