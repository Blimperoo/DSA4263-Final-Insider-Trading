# Datasets Folder 
This folder does not contain our dataset but contains the how to setup your untracked  `data_untracked` folder, to either scrape the necessary datasets from websites or download them through the google drive link. 

## How to setup the `data_untracked` folder
1. Run the below bash code in the root directory of this git repository.
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
            - daily_stock_data_by_ticker/
                - {ticker}.csv...
            - daily_beta_split/
                - part_1.csv
                ...
                - part_5.csv
        - fred/
            - risk_free_rate_daily.csv

    - profile_data/ 

- processed

    - `transactions_final.csv` shape: (3546490, 24)
    - `transactions_abnormal_returns_anomaly_score.csv` rows: 3171001
    - `snorkel_labels.csv`

## Steps to run our code to re-create the dataset