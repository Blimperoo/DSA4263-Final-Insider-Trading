# Datasets Folder 
This folder does not contain our dataset but contains the how to setup your untracked  `data_untracked` folder, to either scrape the necessary datasets from websites or download them through the google drive link. 

## How to setup the `data_untracked` folder
1. Run the below bashcode in the root directory of this git repository.
```{bash}
mkdir -p data_untracked/raw/sec_submissions/interim \
         data_untracked/raw/sec_submissions/compiled \
         data_untracked/raw/abnormal_returns/crsp \
         data_untracked/raw/abnormal_returns/fred \
         data_untracked/raw/profile_data \
         processed
 ```
2. Download the datasets from the google drive link as shown below.
3. Create folders according to diagram below and place the downloaded datasets in the respective folders.

## Folder Structure
- raw/
    - sec_submissions/
        - interim/
        - compiled/
            - NONDERIV_TRANS.csv
            - NONDERIV_HOLDING.csv
            - REPORTINGOWNER.csv
            - DERIV_HOLDING.csv
            - DERIV_TRANS.csv
            - SUBMISSION.csv
            - FOOTNOTES.csv

    - abnormal_returns/
        - crsp/
            - daily_stock_price_by_ticker/
                - {ticker}.csv...
            - crsp_daily_beta.csv
        - fred/
            - treasury_bills.csv
        - transactions_abnormal_returns_{ticker_count}_ticker_COMPARE.csv

    - profile_data/ # Caitlyn can modify this

- processed

    - all_transactions_merged.csv
    - abnormal_returns.csv
    - unique_names_trans_8above.csv
    - unique_ticker_trans_8above.csv


## Steps to run our code to re-create the dataset