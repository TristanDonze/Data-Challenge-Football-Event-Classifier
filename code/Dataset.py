import re
import os
import pandas as pd

def detect_noisy_tweet(tweet) -> bool:
    if re.search(r'https?', tweet):
        return True
    if re.search(r'@.+', tweet):
        return True
    if re.search(r'RT', tweet):
        return True
    return False

def clean_dataframe(df):
    return df[~df['Tweet'].apply(detect_noisy_tweet)]

def clean_dataset(all_matchs):
    all_matchs_clean = {}
    for match, df in all_matchs.items():
        df_clean = clean_dataframe(df)
        all_matchs_clean[f"{match}_clean"] = df_clean
    return all_matchs_clean

def get_data():
    folder_path_train = '../data/train_tweets/'
    folder_path_eval = '../data/eval_tweets/'
    
    all_matchs = {}
    for matchs in os.listdir(folder_path_train):
        if matchs.endswith('.csv'):
            file_path = os.path.join(folder_path_train, matchs)
            all_matchs[matchs[0:-4]] = pd.read_csv(file_path)

    all_matchs_cleaned = clean_dataset(all_matchs)
    
    all_matchs_eval = {}
    for matchs in os.listdir(folder_path_eval):
        if matchs.endswith('.csv'):
            file_path = os.path.join(folder_path_eval, matchs)
            all_matchs_eval[matchs[0:-4]] = pd.read_csv(file_path) 
    
    all_matchs_cleaned_eval = clean_dataset(all_matchs_eval)
    return all_matchs, all_matchs_cleaned, all_matchs_eval, all_matchs_cleaned_eval