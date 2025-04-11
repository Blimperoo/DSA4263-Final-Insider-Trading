import pandas as pd
import numpy as np
import re
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_location import folder_location

PROCESSED_DATA_FOLDER = folder_location.PROCESSED_DATA_FOLDER
ABNORMAL_CSV = folder_location.ABNORMAL_CSV

FEATURES_FOLDER = folder_location.FEATURES_DATA_FOLDER
FINAL_FILE = "other_feature.csv"

def create_features():
    """This function will create the key-feature graph if file is not found and then return this Datafram

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FEATURES_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Other Features Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FEATURES_FOLDER}/{FINAL_FILE}')
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
        ).replace([np.inf, -np.inf], np.nan)

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
        # security category
        ##############################
        trans_score_df['trans_amt'] = trans_score_df['TRANS_SHARES'] * trans_score_df['TRANS_PRICEPERSHARE']
        def categorize_security(desc):
            """
            Simple function to bucket security descriptions into:
            - 'common_stock' 
            - 'derivative'
            - 'other'
            """
            desc_lower = str(desc).lower()
            if ("com" in desc_lower and "stoc" in desc_lower) or "class a" in desc_lower or "ordinary shares" in desc_lower:
                # check again
                return "common_stock"
            #elif any(keyword in desc_lower for keyword in ["option", "warrant", "right to buy"]):
            #    #need to double check for options (ref to brashears)
            #    return "derivative"
            else:
                return "other"

        trans_score_df["security_category"] = trans_score_df["SECURITY_TITLE"].apply(categorize_security)

        ##############################
        # nature_of_ownership
        ##############################
        # Precompiled regex pattern
        PATTERN_401K_PLAN = re.compile(r'(?i)401\s*[-\s]*[\(\)]*\s*k\s*[\(\)]*(\s*plan)?')
        PATTERN_SPACE_AFTER_401K = re.compile(r'(?i)(401k)([a-z])')
        PATTERN_SPOUSE_401K = re.compile(r'(?i)(spouse)(401k)')
        PATTERN_ESOP = re.compile(r'(?i)\b[a-z]*esop\b')
        PATTERN_CLEAN = re.compile(r'[^a-z0-9 ]')

        def map_nature_of_ownership_score(text):
            """
            Maps the NATURE_OF_OWNERSHIP field into a numeric signal score (0–5).

            Steps:
            1. Normalizes variations of "401(k)" (allowing hyphens, spaces, parentheses, or an optional "plan")
                using precompiled regex patterns.
            2. Inserts a space if "401k" is attached to letters (e.g., "401kplan" -> "401k plan").
            3. Normalizes ESOP variants (e.g., 'paesop', 'xaesop') to "esop".
            4. Cleans the text by removing punctuation (non-alphanumeric except spaces), lowercases, and tokenizes.
            5. Checks for multi-word phrases (searched in the entire cleaned text) and for single tokens 
                (searched as substrings within each token) against a mapping.
            6. Returns the highest score found.
            """
            if pd.isna(text):
                return 0
            
            text = text.strip()

            # Normalize 401(k) variants:
            text = PATTERN_401K_PLAN.sub(lambda m: "401k plan" if m.group(1) and m.group(1).strip().lower() == "plan" else "401k", text)
            text = PATTERN_SPACE_AFTER_401K.sub(r'\1 \2', text)
            text = PATTERN_SPOUSE_401K.sub(r'\1 \2', text)
            
            # Normalize ESOP variants to "esop"
            text = PATTERN_ESOP.sub('esop', text)
            
            # Clean text: remove non-alphanumeric (except spaces), lowercase, and tokenize.
            clean_text = PATTERN_CLEAN.sub(' ', text.lower())
            tokens = clean_text.split()

            # Define mapping groups: each tuple is (list of keywords/phrases, score)
            mapping = [
                (["direct", "self", "own", "sole"], 5),                   # Direct ownership
                (["spouse", "wife", "husband", "joint", "son", "daughter", "child"], 4),  # Family-related
                (["ira", "401k", "401k plan", "retirement", "custodian", "esop"], 3),       # Retirement accounts
                (["revocable trust", "living trust", "trustee", "trust"], 2),  # Revocable trusts
                (["irrevocable", "grat", "foundation", "charitable"], 1),   # Complex/charitable
                (["llc", "lp", "corp", "company", "fund", "partners"], 0),    # Entity-level holding
                (["footnote", "note"], 0),                                  # Vague/unclear
            ]
            
            scores = []
            for keywords, score in mapping:
                for keyword in keywords:
                    if " " in keyword:
                        # For multi-word phrases, check if the entire phrase is in the cleaned text.
                        if keyword in clean_text:
                            scores.append(score)
                            break
                    else:
                        # For single tokens, check if any token contains the keyword as a substring.
                        if any(keyword in token for token in tokens):
                            scores.append(score)
                            break

            if not scores:
                return 0
            
            return max(scores)

        # Example usage: applying to the DataFrame column.
        trans_score_df['beneficial_ownership_score'] = trans_score_df['NATURE_OF_OWNERSHIP'].apply(map_nature_of_ownership_score)

        ##############################
        # Title Score ('RPTOWNER_TITLE_#')
        ##############################
        def is_pure_na_or_note(cell):
            """
            Returns True if the cell is:
            - a missing value (NaN), or
            - a string that, when split by '#', consists entirely of tokens that are either "nan"
                or a note indicator (e.g., tokens that start with "see remarks" or "see footnote").
            """
            if pd.isna(cell):
                return True
            
            # Split the cell text by '#' and examine each token.
            tokens = str(cell).split('#')
            for token in tokens:
                token_clean = token.strip()
                # Convert token to lowercase and remove any leading asterisks.
                token_lower = token_clean.lower().lstrip('*').strip()
                # Remove enclosing parentheses if present.
                if token_lower.startswith('(') and token_lower.endswith(')'):
                    token_lower = token_lower[1:-1].strip()
                if token_lower == 'nan':
                    continue
                # Allow any token that starts with "see remarks" or "see footnote"
                if token_lower.startswith("see remarks") or token_lower.startswith("see footnote"):
                    continue
                # Found a token that is not NA or a note indicator.
                return False
            return True

        #seniority mapping
        seniority_keywords = {
            "lead independent director": 3,
            "chairperson": 3,
            "chief executive officer": 3,
            "ceo": 3,
            "coo": 3,
            "president": 2,
            "cfo": 2,
            "managing director": 2,
            "evp":2,
            "vp": 2,
            "vice president": 2,
            "officer": 2,
            "chairman": 2,
            "treasurer": 1,
            "board": 1,
            "director": 1,
            "secretary": 1,
        }

        # Precompile a regex pattern to clean non-alphanumeric characters (except spaces).
        PATTERN_CLEAN = re.compile(r'[^a-z0-9 ]')

        def map_title_score(text):
            """
            Maps the title text to a numeric title score based on seniority_keywords.
            
            Process:
            1. If the text is missing or purely a note/NA indicator (using is_pure_na_or_note), return 0.
            2. Clean the text by removing non-alphanumeric characters (except spaces) and converting to lowercase.
            3. Tokenize the cleaned text.
            4. For each keyword in the seniority_keywords mapping:
                - If the keyword contains a space (multi-word phrase), check if it appears in the entire cleaned text.
                - Else, check if any token contains the keyword as a substring.
            5. Return the total score
            """
            if is_pure_na_or_note(text):
                return 0

            text = text.strip()
            # Clean the text: remove punctuation and extra characters, then lowercase.
            clean_text = PATTERN_CLEAN.sub(' ', text.lower())
            tokens = clean_text.split()

            total_score = 0
            for keyword, score in seniority_keywords.items():
                if " " in keyword:
                    # For multi-word keyword, search the entire clean_text.
                    if keyword in clean_text:
                        total_score += score
                else:
                    # For single-word keywords, search each token.
                    if any(keyword in token for token in tokens):
                        total_score += score

            return total_score
        
        trans_score_df['title_score'] = trans_score_df['RPTOWNER_TITLE_#'].apply(map_title_score)

        ##############################
        # Features based on transaction, execution and filing date
        ##############################

        # 3 categories according to SEC 
        trans_score_df['TRANS_TIMELINESS_clean'] = trans_score_df['TRANS_TIMELINESS'].replace({'E': 'early', 'L': 'late', np.nan: 'on_time'})

        # Note that there are transactions deemed to be executed more than a year before/after. Only 3 categories
        trans_score_df['execution_timeliness'] = (trans_score_df['DEEMED_EXECUTION_DATE'] - trans_score_df['TRANS_DATE']).apply(lambda x: 'before_trans' if x.days < 0 else 'after_trans' if x.days > 0 else 'on_trans')

        # Compute lag in days
        trans_score_df['filing_lag_days'] = (trans_score_df['FILING_DATE'] - trans_score_df['TRANS_DATE']).dt.days

        # Compute days categorical label
        trans_score_df['filing_timeliness'] = trans_score_df['filing_lag_days'].apply( 
            lambda x: 'early_filing' if pd.notna(x) and x < 0
                else ('on_time_filing' if pd.notna(x) and x == 0
                       else ('late_filing' if pd.notna(x) and x > 0
                              else 'missing_filing_date'))
        ) 
        ##############################
        # Save file
        ##############################
        features_to_keep = ["net_trading_intensity", "net_trading_amt", "relative_trade_size_to_self", 
                            "relative_trade_size_to_others", "trans_amt", "security_category","beneficial_ownership_score","title_score",
                            "TRANS_TIMELINESS_clean", 'execution_timeliness', 'filing_lag_days', 'filing_timeliness']
        
        key = ["ACCESSION_NUMBER", "TRANS_SK"]

        df_to_save = trans_score_df[features_to_keep + key]
        df_to_save.to_csv(f'{FEATURES_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save

    return df_to_return