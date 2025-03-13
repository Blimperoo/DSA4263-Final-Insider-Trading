import os
import re
import requests
import zipfile
import shutil
import glob
import pandas as pd
from bs4 import BeautifulSoup
import json

# -------------------------------
# Scrape and Download Zip Files
# -------------------------------
raw_data_folder = "data/raw"
if not os.path.exists(raw_data_folder):
    os.makedirs(raw_data_folder)
    print(f"Folder '{raw_data_folder}' created.")
else:
    print(f"Folder '{raw_data_folder}' already exists.")

# URL and headers for SEC data
url = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
headers = {"User-Agent": "DSA4263 (dsa4263hi@gmail.com)"}

response = requests.get(url, headers=headers)
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
                if year > 2020:      ##CHANGE THIS 
                    zip_links.append(href)
            except ValueError:
                pass
        else:
            print(f"No year found in URL: {href}")
print(f"Found {len(zip_links)} zip file links")

# Download each zip file
for link in zip_links:
    try:
        r = requests.get(link, headers=headers)
        r.raise_for_status()
    except Exception as e:
        print(f"Error downloading {link}: {e}")
        continue

    zip_filename = os.path.join(raw_data_folder, link.split("/")[-1])
    with open(zip_filename, "wb") as f:
        f.write(r.content)
    print(f"Downloaded: {zip_filename}")

# -------------------------------
# Extract and Merge TSV Files
# -------------------------------

#temp folder 
temp_extracted = "temp_extracted"
# Final folder for merged output
final_folder = "data/interim"
os.makedirs(final_folder, exist_ok=True)

# Dictionary to store DataFrames keyed by the TSV filename (e.g., "DERIV_HOLDING.tsv")
merged_data = {}

# Keep track of whether we've copied the metadata and readme files yet
metadata_copied = 0

# Process each downloaded zip file from the RAW_DATA folder
zip_files = glob.glob(os.path.join(raw_data_folder, "*.zip"))

for zip_path in zip_files:
    
    #temp file
    if os.path.exists(temp_extracted):
        shutil.rmtree(temp_extracted)
    os.makedirs(temp_extracted, exist_ok=True)

    # Extract the zip contents to temp
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_extracted)

    # Process extracted files: merge TSVs and copy metadata/readme
    for root, dirs, files in os.walk(temp_extracted):
        for filename in files:
            filepath = os.path.join(root, filename)
            # If it's a TSV, load and merge it
            if filename.lower().endswith(".tsv"):
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
            elif filename in ["FORM_345_metadata.json", "FORM_345_readme.htm"]:
                if metadata_copied < 2:
                    dest_path = os.path.join(final_folder, filename)
                    shutil.copy2(filepath, dest_path)
                    print(f"Copied metadata/readme: {filename}")
                    metadata_copied += 1

    # Remove the temporary extraction folder
    shutil.rmtree(temp_extracted, ignore_errors=True)

# Write out the merged TSV files into the final folder
for tsv_name, df in merged_data.items():
    output_path = os.path.join(final_folder, tsv_name)
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Merged TSV saved: {output_path}")

print("All TSV files have been merged.")

 

# -----------------------------------------------------------------------------
# Load metadata and build a mapping from TSV filename -> {col_name -> datatype}
# -----------------------------------------------------------------------------
with open("data/interim/FORM_345_metadata.json", "r") as f:
    metadata = json.load(f)

conversion_mapping = {}
for table in metadata["tables"]:
    tsv_filename = table["url"]  # e.g. "OWNER_SIGNATURE.tsv"
    col_mappings = {}
    for col in table["tableSchema"]["columns"]:
        # e.g. col["name"] might be "ACCESSION_NUMBER"
        col_mappings[col["name"]] = col["datatype"]
    conversion_mapping[tsv_filename] = col_mappings

# -----------------------------------------------------------------------------
# Helper function for converting datatypes 
# -----------------------------------------------------------------------------
def convert_value(series, datatype):
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

# -----------------------------------------------------------------------------
# Process each TSV file: read, convert, then save as CSV & delete TSV
# -----------------------------------------------------------------------------
data_folder = "data/interim"
tsv_files = glob.glob(os.path.join(data_folder, "*.tsv"))
problems = []

for tsv_file in tsv_files:
    df = pd.read_csv(tsv_file, sep="\t", dtype=str,low_memory=False) #low_memory params used here for accuracy 
    filename = os.path.basename(tsv_file)
    print(f"\nProcessing {filename}...")

    #Debugging
    #print("  Columns found in TSV:", df.columns.tolist())

    # Conversion of datatypes 
    if filename in conversion_mapping:
        meta_for_file = conversion_mapping[filename]
        for col_name, datatype_info in meta_for_file.items():
            if col_name in df.columns:
                df[col_name] = convert_value(df[col_name], datatype_info)
            else:
                problems.append(f"Column '{col_name}' not found in {filename}.")
                #print(f"Warning: Column '{col_name}' not found in {filename}.")
    else: 
        print(f"  No metadata mapping found for {filename}.")

    # Debugging
    # print("  Data types after conversion:")
    # print(df.dtypes)

    #Store as csv files
    csv_filename = os.path.splitext(tsv_file)[0] + ".csv"
    df.to_csv(csv_filename, index=False)
    print(f"  Saved converted data to {csv_filename}")

    #Delete original tsv
    os.remove(tsv_file)
    print(f"  Deleted original file: {tsv_file}")





