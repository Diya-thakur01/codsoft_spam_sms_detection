# src/preprocessing.py

import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='latin-1')

    # Keep only necessary columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels: ham = 0, spam = 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df['message'], df['label']
