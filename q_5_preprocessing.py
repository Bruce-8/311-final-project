import re
import pandas as pd

# Load the CSV file, skipping the top row
data = pd.read_csv('csc311 project data - cleaned_data_combined_modified.csv', skiprows=1)

def remove_words(words: list[str], ting: str):
        for word in words:
            ting = re.sub(r'\b' + word + r'\b', '', ting)
        return ting

def process_string(value: str) -> str:
    # Convert the value to lowercase
    value = str(value).lower()
    # Remove all generic words
    value = remove_words(['the', 'an', 'and', 'of', 'to'], value)
    # Remove all s after a word
    value = re.sub(r's\b', '', value)
    # Remove all characters that are not consonants
    value = re.sub(r'[^bcdfghjklmnpqrstvwxyz]', '', value)
    # Combine data points related to "no movie" to data points related to "none"
    if value == 'nn':
        value = 'nmv'
    return value

def get_big_buckets(df: pd.DataFrame) -> dict:
    tings = {}
    specifics = {}

    def add_to_tings(tingz: dict, ting: str, original: str = None):
        if original:
            if ting in specifics:
                specifics[ting].append(original)
            else:
                specifics[ting] = [original]

        if ting in tingz:
            tingz[ting] += 1
        else:
            tingz[ting] = 1

    def remove_words(words: list[str], ting: str):
        for word in words:
            ting = re.sub(r'\b' + word + r'\b', '', ting)
        return ting

    # Loop through the values of the second column and print each one
    for value in df.iloc[:, 1]:
        add_to_tings(tings, process_string(value), value)

    bigtings = {}
    bigenough = 15
    for ting in tings:
        if tings[ting] > bigenough:
            bigtings[ting] = tings[ting]
    
    return bigtings

def q_5_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Drop the columns that are not needed
    buckets = {
        'cldywthchncmtbll',
        'nvm',
        'hmln',
        'spdrmn',
        'tngmtntnnjtrtl',
        'rttll',
        'vngr',
        'fndngnm',
        'rshhr',
        'dcttr',
        'lddn',
        'sprtdwy',
        'jrdrmssh',
        'mnstrnc',
    }
    
    def bucket_map(value: str) -> str:
        value = process_string(value)
        if value in buckets:
            return value
        return ''

    df['big_bucket'] = df.iloc[:, 1].map(bucket_map)
    return df
