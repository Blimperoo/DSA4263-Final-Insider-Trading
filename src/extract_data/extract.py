import numpy as np
import pandas as pd
import sys
import os
import re
import requests
import zipfile
import shutil
import glob
import pandas as pd
from bs4 import BeautifulSoup
import json

from path_location import folder_location

script_dir = os.path.dirname(os.path.abspath(__file__))

if script_dir not in sys.path:
    sys.path.append(script_dir)

parent_dir = os.path.dirname(os.path.abspath(f'{__file__}/..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

RAW_DATA_FOLDER = folder_location.SEC_SUBMISSIONS_FOLDER
PROCESSED_FOLDER = folder_location.PROCESSED_DATA_FOLDER

# URL for SEC data
URL = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"

# Headers for SEC data
HEADERS = {"User-Agent": "DSA4263 (dsa4263@gmail.com)"}

# Match with little sis network data
YEARS_THRESHOLD = (2005, 2024)

TEMP_EXTRACTED = RAW_DATA_FOLDER+"/temp_extracted"

FINAL_FOLDER = RAW_DATA_FOLDER+"/compiled"

# Large files to exclude
EXCLUDED_FILES = ['owner_signature.tsv', 'footnotes.tsv']

# Final files present in compiled
COMPILED_FILES = ["DERIV_HOLDING.csv", "DERIV_TRANS.csv", "FOOTNOTES.csv", "NONDERIV_HOLDING.csv",\
                  "NONDERIV_TRANS.csv", "REPORTINGOWNER.csv", "SUBMISSION.csv"]

class Data_Extractor:
    def __init__(self):
        self.conversion_mapping = {}
        
    def create_form4(self):
        if self.__check_compiled_files():
            print("========== Required SEC files present ==========")
        else:
            print("========== Some SEC files not present ==========")
            print("=============== Begin extracting ===============")
            self.__extract_zip()
            self.__extract_and_merge_tsv_files()
            self.__load_metadata_and_build_mapping()
            self.__process_tsv_files()
    
    def merge_form4(self):
        if self.__check_processed_files():
            print("========== Required merged transactions present ==========")
        else:
            print("========== merged transactions files not present ==========")
            print("=============== Begin merging ===============")
            self.__merge_form4()
        
################################################################################
# CHECK IF COMPILED FILES EXIST
################################################################################

    def __check_compiled_files(self):
        current_compiled_files = os.listdir(FINAL_FOLDER)
        for file in COMPILED_FILES:
            if file not in current_compiled_files:
                return False
        return True
    
    def __check_processed_files(self):
        files = os.listdir(PROCESSED_FOLDER)
        if "transactions_final.csv" not in files:
            return False
        return True


################################################################################
# EXTRACT ZIP FILES FROM SEC 
################################################################################

    def __extract_zip(self):
        response = requests.get(URL, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error fetching page: {response.status_code}")
            exit()

        soup = BeautifulSoup(response.text, "html.parser")
        zip_links = []

        # Look for all links that end with '.zip'
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".zip"):
                # Normalize relative URLs if needed
                if href.startswith("/"):
                    href = "https://www.sec.gov" + href
                # Extract a 4-digit year from the URL and filter (e.g., after 2010)
                year_match = re.search(r"(\d{4})", href)
                if year_match:
                    try:
                        year = int(year_match.group(1))
                        if YEARS_THRESHOLD[0] <= year <= YEARS_THRESHOLD[1]:   
                            zip_links.append(href)
                    except ValueError:
                        pass
                else:
                    print(f"No year found in URL: {href}")
        print(f"Found {len(zip_links)} zip file links")

        # Download each zip file
        for link in zip_links:
            try:
                r = requests.get(link, headers=HEADERS)
                r.raise_for_status()
            except Exception as e:
                print(f"Error downloading {link}: {e}")
                continue

            zip_filename = os.path.join(RAW_DATA_FOLDER, 'interim', link.split("/")[-1])
            with open(zip_filename, "wb") as f:
                f.write(r.content)
            print(f"Downloaded: {zip_filename}")

################################################################################
# EXTRACT AND MERGE TSV FILES
################################################################################

    def __extract_and_merge_tsv_files(self):
        #temp folder 
        TEMP_EXTRACTED = RAW_DATA_FOLDER+"/temp_extracted"
        # Final folder for merged output
        FINAL_FOLDER = RAW_DATA_FOLDER+"/compiled"
        os.makedirs(FINAL_FOLDER, exist_ok=True)

        # Dictionary to store DataFrames keyed by the TSV filename (e.g., "DERIV_HOLDING.tsv")
        merged_data = {}
        # Keep track of whether we've copied the metadata and readme files yet
        metadata_copied = 0

        # Process each downloaded zip file from the RAW_DATA folder
        zip_files = glob.glob(os.path.join(RAW_DATA_FOLDER+"/interim", "*.zip"))

        for zip_path in zip_files:
            
            #temp file
            if os.path.exists(TEMP_EXTRACTED):
                shutil.rmtree(TEMP_EXTRACTED)
            os.makedirs(TEMP_EXTRACTED, exist_ok=True)

            # Extract the zip contents to temp
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(TEMP_EXTRACTED)

            # Process extracted files: merge TSVs and copy metadata/readme
            for root, dirs, files in os.walk(TEMP_EXTRACTED):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    # If it's a TSV, load and merge it
                    if filename.lower().endswith(".tsv"):
                        if filename.lower() not in EXCLUDED_FILES: 
                            try:
                                df = pd.read_csv(filepath, sep="\t", low_memory=False)
                            except Exception as e:
                                print(f"Error reading {filepath}: {e}")
                                continue

                            if filename not in merged_data:
                                merged_data[filename] = df
                            else:
                                merged_data[filename] = pd.concat([merged_data[filename], df], ignore_index=True)

                    # If it's metadata or readme, copy only once
                    elif filename in ["insider_transactions_metadata.json", "insider_transactions_readme.htm"]:
                        if metadata_copied < 2:
                            dest_path = os.path.join(FINAL_FOLDER, filename)
                            shutil.copy2(filepath, dest_path)
                            print(f"Copied metadata/readme: {filename}")
                            metadata_copied += 1

            # Remove the temporary extraction folder
            shutil.rmtree(TEMP_EXTRACTED, ignore_errors=True)

        # Write out the merged TSV files into the final folder
        for tsv_name, df in merged_data.items():
            output_path = os.path.join(FINAL_FOLDER, tsv_name)
            df.to_csv(output_path, sep="\t", index=False)
            print(f"Merged TSV saved: {output_path}")

        print("All TSV files have been merged.")


################################################################################
# LOAD METADATA AND BUILD MAPPING
################################################################################

    def __load_metadata_and_build_mapping(self):
        with open(FINAL_FOLDER+"/insider_transactions_metadata.json", "r") as f:
            metadata = json.load(f)

        for table in metadata["tables"]:
            tsv_filename = table["url"]  # e.g. "OWNER_SIGNATURE.tsv"
            if tsv_filename.lower() in EXCLUDED_FILES:
                continue
            col_mappings = {}
            for col in table["tableSchema"]["columns"]:
                # e.g. col["name"] might be "ACCESSION_NUMBER"
                col_mappings[col["name"]] = col["datatype"]
            self.conversion_mapping[tsv_filename] = col_mappings

# -----------------------------------------------------------------------------
# Helper function for converting datatypes 
# -----------------------------------------------------------------------------
    def __convert_value(self, series, datatype):
        base = datatype["base"].lower()
        if "number" in base:
            # Convert to numeric; non-convertible values become NaN
            return pd.to_numeric(series, errors="coerce")
        elif "date (dd-mon-yyyy)" in base:
            # Convert to datetime using format "DD-MON-YYYY" (e.g. "01-JAN-2020")
            return pd.to_datetime(series, format="%d-%b-%Y", errors="coerce")
        else:
            #Ignore for rest
            return series.astype(str)
        
################################################################################
# PROCESS TSV FILES
################################################################################

    def __process_tsv_files(self):
        tsv_files = glob.glob(os.path.join(FINAL_FOLDER, "*.tsv"))
        problems = []

        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, sep="\t", dtype=str,low_memory=False) #low_memory params used here for accuracy 
            filename = os.path.basename(tsv_file)
            if filename.lower() in EXCLUDED_FILES: 
                continue
            print(f"\nProcessing {filename}...")

            # Conversion of datatypes 
            if filename in self.conversion_mapping:
                meta_for_file = self.conversion_mapping[filename]
                for col_name, datatype_info in meta_for_file.items():
                    if col_name in df.columns:
                        df[col_name] = self.__convert_value(df[col_name], datatype_info)
                    else:
                        problems.append(f"Column '{col_name}' not found in {filename}.")
            else: 
                print(f"  No metadata mapping found for {filename}.")

            # Store as csv files
            csv_filename = os.path.splitext(tsv_file)[0] + ".csv"
            df.to_csv(csv_filename, index=False)
            print(f"  Saved converted data to {csv_filename}")

            # Delete original tsv
            os.remove(tsv_file)
            print(f"  Deleted original file: {tsv_file}")

################################################################################
# EXTRACT ZIP FILES FROM SEC 
################################################################################

    def __merge_form4(self):
        
        print("--- Load all the CSV files ---")
        submission_data = pd.read_csv(f"{FINAL_FOLDER}/SUBMISSION.csv", parse_dates=['FILING_DATE'])
        nonderiv_trans_data = pd.read_csv(f"{FINAL_FOLDER}/NONDERIV_TRANS.csv", parse_dates=['TRANS_DATE'])
        deriv_trans_data = pd.read_csv(f"{FINAL_FOLDER}/DERIV_TRANS.csv", parse_dates=['TRANS_DATE'])
        reporting_owner_data = pd.read_csv(f"{FINAL_FOLDER}/REPORTINGOWNER.csv")

        # Define constants
        YEARS_THRESHOLD = (2005, 2021) 
        SELECTED_TRANSACTION_COLS = ['ACCESSION_NUMBER', 'SECURITY_TITLE', 'TRANS_DATE', 'DEEMED_EXECUTION_DATE', 'TRANS_CODE', 'EQUITY_SWAP_INVOLVED',
                             'TRANS_TIMELINESS', 'TRANS_SHARES', 'TRANS_PRICEPERSHARE', 'TRANS_ACQUIRED_DISP_CD',
                             'SHRS_OWND_FOLWNG_TRANS', 'DIRECT_INDIRECT_OWNERSHIP', 'NATURE_OF_OWNERSHIP']
        SUBMISSION_COLS = ['ACCESSION_NUMBER', 'FILING_DATE', 'PERIOD_OF_REPORT', 'ISSUERCIK', 'ISSUERNAME', 'ISSUERTRADINGSYMBOL']
        
        ##############################
        # Merging Transaction data (vertically)
        ##############################
        print("--- Merge transaction data ---")
        df1 = nonderiv_trans_data[['NONDERIV_TRANS_SK'] + SELECTED_TRANSACTION_COLS].copy().rename(columns={'NONDERIV_TRANS_SK':'TRANS_SK'})
        df2 = deriv_trans_data[['DERIV_TRANS_SK'] + SELECTED_TRANSACTION_COLS].copy().rename(columns={'DERIV_TRANS_SK':'TRANS_SK'})
        all_transaction_data = pd.concat([df1,df2], axis=0, ignore_index=True).reset_index(drop=True)
        if all_transaction_data.shape != (9391301, 14):
            print("Unexpected shape for all_transaction_data: ", all_transaction_data.shape, "but expected: (9391301, 14)")
        
        # Select only transactions from 2005 to 2021
        all_transaction_data = all_transaction_data[(all_transaction_data['TRANS_DATE'].dt.year >= YEARS_THRESHOLD[0]) & (all_transaction_data['TRANS_DATE'].dt.year <= YEARS_THRESHOLD[1])]
        if all_transaction_data.shape != (8176007, 14):
            print("Unexpected shape for all_transaction_data: ", all_transaction_data.shape, "but expected: (8176007, 14)")
        
        # Merge with submission data on ACCESSION_NUMBER 
        all_transaction_data_2 = all_transaction_data.merge(submission_data[SUBMISSION_COLS], on='ACCESSION_NUMBER', how='left')
        
        assert all_transaction_data_2.shape[0] == all_transaction_data.shape[0]

        ##############################
        # Merging reporting owner data to submission 
        ##############################
        print("--- Merge reporting owner data ---")
        # Ensure that ';' is NOT in these columns, such that a split by ';' will give the exact number of elements
        assert reporting_owner_data['RPTOWNERCIK'].map(str).str.contains(';').sum() == 0
        assert reporting_owner_data['RPTOWNERNAME'].str.contains(';').sum() == 0
        assert reporting_owner_data['RPTOWNER_RELATIONSHIP'].str.contains(';').sum() == 0

        # Ensure that '#' is NOT in this columns, Note that ';' is present in too many of these rows
        reporting_owner_data['RPTOWNER_TITLE'] = reporting_owner_data['RPTOWNER_TITLE'].astype(str).str.replace('#', '', regex=False)
        assert reporting_owner_data['RPTOWNER_TITLE'].str.contains('#').sum() == 0

        # One form4 submission can have multiple reporting owners. Join them together below. 
        def join_with_nan(series):
            '''catch NaNs and convert to 'NaN' string, to ensure they are not removed and the count is correct'''
            return ';'.join(series.fillna('NaN').astype(str))
        def hashtag_join_with_nan(series):
            '''catch NaNs and convert to 'NaN' string, to ensure they are not removed and the count is correct'''
            return '#'.join(series.fillna('NaN').astype(str))

        # Group by ACCESSION_NUMBER and join other fields as semicolon-separated strings
        grouped_reporting_owner = reporting_owner_data.groupby('ACCESSION_NUMBER', as_index=False).agg({
            'RPTOWNERCIK': [join_with_nan, 'count'],
            'RPTOWNERNAME': join_with_nan,
            'RPTOWNER_RELATIONSHIP': join_with_nan,
            'RPTOWNER_TITLE': hashtag_join_with_nan
        })

        assert reporting_owner_data['ACCESSION_NUMBER'].nunique() == grouped_reporting_owner.shape[0]

        # Flatten MultiIndex columns
        grouped_reporting_owner.columns = ['ACCESSION_NUMBER', 'RPTOWNERCIK_;', 'NUM_RPTOWNERCIK', 'RPTOWNERNAME_;', 'RPTOWNER_RELATIONSHIP_;','RPTOWNER_TITLE_#']

        all_transaction_direct_final = all_transaction_data_2.merge(grouped_reporting_owner, on='ACCESSION_NUMBER', how='left') 

        assert all_transaction_direct_final.shape[0] == all_transaction_data_2.shape[0]

        ##############################
        # Filter out codes
        ##############################
        print("--- Filter out codes ---")
        all_transaction_direct_final_filter = all_transaction_direct_final[all_transaction_direct_final['TRANS_CODE'].isin(['P', 'S', 'J', 'V','I','G'])]
        if all_transaction_direct_final_filter.shape != (3546490, 24):
            print("Unexpected shape for all_transaction_direct_final_filter: ", all_transaction_direct_final_filter.shape, "but expected: (3546490, 24)")

        ##############################
        # Export to csv
        ##############################
        print(f"--- Export final merged file, shape {all_transaction_direct_final_filter.shape} to csv ---")
        all_transaction_direct_final_filter.to_csv(f"{PROCESSED_FOLDER}/transactions_final.csv", index=False)