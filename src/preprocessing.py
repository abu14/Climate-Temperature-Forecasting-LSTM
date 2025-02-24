import pandas as pd
import tensorflow as tf
import os

def load_jena_data():
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path = zip_path.replace('.zip', '')
    return pd.read_csv(os.path.join(csv_path, 'jena_climate_2009_2016.csv'))

def clean_data(df):
    df.columns = df.columns.map(lambda x : x.replace(' ', '_').replace('.',''))
    df.index = pd.to_datetime(df['Date_Time'], format='%d.%m.%Y %H:%M:%S')
    return df.drop('Date_Time', axis=1)

def train_val_test_split(df):
    train_split = int(len(df)* 0.7)
    validation_split = int(len(df)* 0.1)
    
    train_df = df[:train_split]
    validation_df = df[train_split: validation_split + train_split]
    test_df = df[validation_split + train_split:]
    
    # Track original lengths for alignment
    return (train_df, validation_df, test_df), {
        'original_lengths': {
            'train': len(train_df),
            'val': len(validation_df),
            'test': len(test_df)
        }
    }