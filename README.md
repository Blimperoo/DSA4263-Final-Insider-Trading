# Datasets Folder 
This folder does not contain our dataset but contains the how to setup your untracked  `data_untracked` folder, to either scrape the necessary datasets from websites or download them through the google drive link. 

## How to setup the `data_untracked` folder
1. Run the below bash code in the root directory of this git repository.
```{bash}
mkdir -p data_untracked/raw/sec_submissions/interim \
         data_untracked/raw/sec_submissions/compiled \
         data_untracked/raw/abnormal_returns/crsp \
         data_untracked/raw/abnormal_returns/fred \
         data_untracked/raw/abnormal_returns/trans_ars \ 
         data_untracked/raw/profile_data \
         data_untracked/processed \ 
         data_untracked/features \
 ```
2. Download the datasets from the google drive link as shown below: 
    ### UPDATE HERE 

3. Create folders according to diagram below and place the downloaded datasets in the respective folders.
   ```
         data_untracked/
         ├── raw/
         │   ├── sec_submissions/
         │   │   ├── interim/                          # Raw Form 4 filings (zipped), 2005–2024
         │   │   │   ├── 2015q1_form345.zip
         │   │   │   ├── ...
         │   │   │   └── 2024q4_form345.zip
         │   │   └── compiled/                         # Compiled SEC submission data
         │   │       ├── NONDERIV_TRANS.csv
         │   │       ├── NONDERIV_HOLDING.csv
         │   │       ├── REPORTINGOWNER.csv
         │   │       ├── DERIV_HOLDING.csv
         │   │       ├── DERIV_TRANS.csv
         │   │       ├── SUBMISSION.csv
         │   │       └── FOOTNOTES.csv
         |   |
         │   ├── abnormal_returns/
         │   │   ├── crsp/
         │   │   │   ├── daily_stock_data_by_ticker/   # Daily stock prices split by ticker
         │   │   │   │   ├── AAPL.csv
         │   │   │   │   ├── MSFT.csv
         │   │   │   │   └── ...
         │   │   │   └── daily_beta_split/             # Batched beta calculation data
         │   │   │       ├── part_1.csv
         │   │   │       ├── ...
         │   │   │       └── part_5.csv
         │   │   ├── fred/
         │   │   │   └── risk_free_rate_daily.csv      # FRED risk-free rate data
         │   │   └── trans_ars/                        # Abnormal return calculations (batched)
         │   │       └── ...
         │   |
         │   └── profile_data/
         │       ├── adjacency_list.csv                 # Adjacency list of the network   
         │       ├── congress_date_subcomm_mapper.pkl           
         │       ├── house_membership_by_date.pkl
         │       ├── tic_to_subcomm_mapper.pkl
         │       ├── TIC to SIC.xlsx
         │       ├── house.csv
         │
         ├── processed/
         │   ├── transactions_final.csv                                # Merged transaction data from SEC Form 4 (3546490, 24)
         │   ├── transactions_abnormal_returns.csv                     # Transactions with calculated abnormal returns (3171001, 46)
         │   ├── transactions_abnormal_returns_anomaly_score.csv       # Adds anomaly score partial labels (3171001, 83)
         │   ├── merged_txns_SNORKEL.csv                               # Adds labels from Snorkel labeling functions
         │   ├── name_matched_improved.csv                             # Form 4 to LittleSis Name Match
         │   ├── merged_relationships_full.csv                         # Network data
         │   └── ground_truth_matching_keys.csv                        # Ground truth transactions caught and flagged by SEC
         │
         ├── features/                                                 
         │   ├── footnote_word_count_feature.csv
         │   ├── other_feature.csv
         │   ├── transaction_code.csv
         │   └── graph_feature.csv
```
## Steps to run our code to re-create the dataset
