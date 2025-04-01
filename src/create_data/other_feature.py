import pandas as pd
import numpy as np
import os

PROCESSED_DATA_FOLDER = "../data_untracked/processed"
ABNORMAL_CSV = "snorkel_labels.csv"

FINAL_FOLDER = "../data_untracked/features"
FINAL_FILE = "other_feature.csv"

def create_features():
    """This function will create the key-feature graph if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FINAL_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Other Features Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
    else: # Create features and save
        print("=== Other Features Key file not found, begin creating  ===")
        
        trans_score_df = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{ABNORMAL_CSV}', parse_dates=['TRANS_DATE'])

        # ensure no NA values and get initial num_rows
        assert trans_score_df['TRANS_ACQUIRED_DISP_CD'].isna().sum() == 0
        num_rows = trans_score_df.shape[0]

        ##############################
        # net_trading_intensity, net_trading_amt
        ##############################
        
        ## Aggregate to firm-date level and create a new dataframe ni_df
        trans_score_df['is_buy'] = trans_score_df['TRANS_ACQUIRED_DISP_CD'] == 'A'
        trans_score_df['is_sell'] = trans_score_df['TRANS_ACQUIRED_DISP_CD'] == 'D'
        ni_df = (
            trans_score_df
            .groupby(['ISSUERTRADINGSYMBOL', 'TRANS_DATE'])
            .agg(buy_count=('is_buy', 'sum'),
                sell_count=('is_sell', 'sum'),
                buy_amt=('trans_amt', lambda x: x[trans_score_df.loc[x.index, 'is_buy']].sum()),
                sell_amt=('trans_amt', lambda x: x[trans_score_df.loc[x.index, 'is_sell']].sum()))
            .reset_index()
        )

        ni_df['net_trading_intensity'] = (
            (ni_df['buy_count'] - ni_df['sell_count']) /
            (ni_df['buy_count'] + ni_df['sell_count'])
        )

        ni_df['net_trading_amt'] = (
            (ni_df['buy_amt'] - ni_df['sell_amt']) /
            (ni_df['buy_amt'] + ni_df['sell_amt'])
        ).replace([np.inf, -np.inf], np.nan)

        ## Merge back to the original dataframe
        trans_score_df = trans_score_df.merge(
            ni_df[['ISSUERTRADINGSYMBOL', 'TRANS_DATE', 'net_trading_intensity', 'net_trading_amt']],
            on=['ISSUERTRADINGSYMBOL', 'TRANS_DATE'],
            how='left'
        )
        
        assert trans_score_df.shape[0] == num_rows

        ##############################
        # relative_trade_size_to_self
        ##############################
        trans_score_df.sort_values(by=['RPTOWNERCIK', 'TRANS_DATE'], inplace=True)
        def calc_group_avg_past(x):
            return x.expanding().mean().shift(1)

        trans_score_df['avg_trans_shares_past'] = (
            trans_score_df.sort_values(['RPTOWNERCIK', 'TRANS_DATE'])
                            .groupby('RPTOWNERCIK')['TRANS_SHARES']
                            .transform(calc_group_avg_past)
        )

        trans_score_df['relative_trade_size_to_self'] = (
            trans_score_df['TRANS_SHARES'] / trans_score_df['avg_trans_shares_past']
        )

        # Fill NaNs with 1, since it is the first ever 'standard' value to expect from a person? 
        trans_score_df['relative_trade_size_to_self'].fillna(1, inplace=True)

        assert trans_score_df.shape[0] == num_rows

        ##############################
        # relative_trade_size_to_others
        ##############################
        trans_score_df['market_value_traded'] = trans_score_df['VOL'] * trans_score_df['PRC'].abs()
        trans_score_df['relative_trade_size_to_others'] = (trans_score_df['trans_amt'] / trans_score_df['market_value_traded']
                                                           ).replace([np.inf, -np.inf], np.nan)

        ##############################
        # trans_amt
        ##############################
        trans_score_df['trans_amt'] = trans_score_df['TRANS_SHARES'] * trans_score_df['TRANS_PRICEPERSHARE']

        ##############################
        # Save file
        ##############################
        features_to_keep = ["net_trading_intensity", "net_trading_amt", "relative_trade_size_to_self", "relative_trade_size_to_others", "trans_amt"]
        key = ["ACCESSION_NUMBER", "TRANS_SK"]

        df_to_save = trans_score_df[features_to_keep + key]
        df_to_save.to_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save

    return df_to_return