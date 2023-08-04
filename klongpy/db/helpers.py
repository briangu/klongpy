import gzip
from io import BytesIO

import pandas as pd


def save_df(file_path, df):
    with open(file_path, "wb") as f:
        with gzip.open(f, "wb", compresslevel=9) as gzf:
            df.to_pickle(gzf)


def read_df(file_path):
    with open(file_path, "rb") as f:
        with gzip.open(f, "rb") as gzf:
            return pd.read_pickle(gzf)


def df_memory_usage(df):
    return df.memory_usage(deep=True).sum()


def serialize_gzip_df(df):
    bio = BytesIO()
    with gzip.open(bio, 'wb') as f:
        df.to_pickle(f)
    return bio.getvalue()


def deserialize_gzip_df(data):
    bio = BytesIO(data)
    with gzip.open(bio, "rb") as gzf:
        return pd.read_pickle(gzf)


def serialize_df(df):
    bio = BytesIO()
    df.to_pickle(bio)
    return bio.getvalue()


def deserialize_df(data):
    bio = BytesIO(data)
    return pd.read_pickle(bio)
