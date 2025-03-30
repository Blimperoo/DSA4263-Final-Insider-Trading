import pandas as pd
import numpy as np
import re
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

PROCESSED_DATA_FOLDER = "../data_untracked/processed"
ABNORMAL_CSV = "snorkel_labels.csv"

SEC_DATA_FOLDER = "../data_untracked/raw/sec_submissions/compiled"
FOOTNOTE_FILE = "FOOTNOTES.csv"

FINAL_FOLDER = "../data_untracked/features"
FINAL_FILE = "footnote_word_count_feature.csv"

# Lemmatizer used in tagging and lemmatizing words
LEMMATIZER = WordNetLemmatizer()

# Stop words to remove common words that appear
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["shall", "should"])


def create_features():
    """This function will create the key-feature footnote if file is not found and then return this Dataframe

    Returns:
        Dataframe: of features
    """
    current_compiled_files = os.listdir(FINAL_FOLDER)
    current_compiled_sec_submissions = os.listdir(SEC_DATA_FOLDER)
    
    # Checks if the file is found
    if FINAL_FILE in current_compiled_files:
        print("=== Footnote Key file is found. Extracting ===")
        df_to_return = pd.read_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
    
    # Creates the footnote compilation
    elif FOOTNOTE_FILE in current_compiled_sec_submissions:
        print(f"=== Footnote Key file not found, begin extracting {FOOTNOTE_FILE} ===")
        
        footnotes = pd.read_csv(f'{SEC_DATA_FOLDER}/{FOOTNOTE_FILE}')
        
        ##############################
        # Merging footnotes together for the same ACCESSION_NUMBER
        ##############################
        accession_number_in_final = pd.read_csv(f'{PROCESSED_DATA_FOLDER}/{ABNORMAL_CSV}')[["ACCESSION_NUMBER"]].drop_duplicates()
        acc = accession_number_in_final["ACCESSION_NUMBER"]
        
        
        print("=== Merging footnotes ===")
        df_footnote = footnotes.copy()
        df_footnote = df_footnote[df_footnote["ACCESSION_NUMBER"].isin(acc)]
        df_footnote = df_footnote.sort_values(by=["ACCESSION_NUMBER", "FOOTNOTE_ID"])
        df_footnote["FOOTNOTE_TXT"] = df_footnote["FOOTNOTE_TXT"].astype(str)
        df_grouped = df_footnote.groupby("ACCESSION_NUMBER", sort=True)["FOOTNOTE_TXT"].apply(lambda x: " ".join(x)).reset_index()
        
        
        ##############################
        # Count instances of words
        ##############################
        
        text_list = ["distribution", "sell", "trading"]
        text_code = ["10b5-1", "16b-3"]
        print("=== create processed footnotes ===")
        df_grouped["processed"] = df_grouped["FOOTNOTE_TXT"].apply(lambda row: preprocess_text(row))
        
        for text in text_list:
            print(f"Creating feature for {text}")
            df_grouped[text] = df_grouped["processed"].str.count(preprocess_text(text))
        
        for code in text_code:
            print(f"Creating feature for {code}")
            df_grouped[code] = df_grouped["FOOTNOTE_TXT"].str.count(code)
        
        ##############################
        # Save key file
        ##############################
        
        features_to_keep = text_list + text_code
        key = ["ACCESSION_NUMBER"]
        
        df_to_save = df_grouped[features_to_keep + key]
        df_to_save = pd.merge(accession_number_in_final, df_to_save, left_on="ACCESSION_NUMBER", right_on="ACCESSION_NUMBER", how="left")
        
        for feature in features_to_keep:
            df_to_save[feature] = df_to_save[feature].fillna(0)
        
        df_to_save.to_csv(f'{FINAL_FOLDER}/{FINAL_FILE}')
        df_to_return = df_to_save
        
        

        
    else: # Create features and save
        print(f"=== {FOOTNOTE_FILE} not found!! ===")
        df_to_return = pd.DataFrame()

    return df_to_return

def preprocess_text(text):
    """Cleans a text by uncapitalizing, removing words and lemmatizing them

    Args:
        text (str): A sentence of words

    Returns:
        str: A cleaned sentence
    """
    text = text.lower()
    removed_unecessary_text = remove_words(text)
    lemmatized_text = lemmatize_text(removed_unecessary_text)

    return lemmatized_text


def get_wordnet_pos(nltk_tag):
    """ This function attempts to categorize the words that comes in

    Args:
        nltk_tag (str): A string that will be tagged

    Returns:
        str: A part of speech tagged
    """
    if nltk_tag.startswith("J"):  # Adjective
        return wordnet.ADJ
    elif nltk_tag.startswith("N"):  # Noun
        return wordnet.NOUN
    elif nltk_tag.startswith("V"):  # Verb
        return wordnet.VERB
    elif nltk_tag.startswith("R"):  # Adverb
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if unknown
    
def lemmatize_text(text):
    """ This function will lemmatize words to their base form: forced -> force

    Args:
        text (str): A sentence of words 

    Returns:
        str: A sentence that has words reduced to lemmatized form
    """
    words = word_tokenize(text)  # Tokenize text
    tagged_words = pos_tag(words)  # POS tagging

    lemmatized_words = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_words if LEMMATIZER.lemmatize(word, get_wordnet_pos(tag)) not in  STOP_WORDS
    ]

    return " ".join(lemmatized_words)  # Reconstruct sentence


def remove_words(text):
    """Remove numbers, months, days and unnecessary text

    Args:
        text (str): A sentence of words

    Returns:
        str: Trimmed sentence, removing unnecessary stuff
    """
    text = re.sub(r"[0-9,.%\$\"\(\)\/_]+", "", text)
    text = re.sub(r"(january|february|march|april|may|june|july|august|september|october|november|december|\
                  monday|tuesday|wednesday|thursday|friday)", "", text)
    return text
