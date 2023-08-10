import gzip
import logging
import os
import pickle
import threading
from io import BytesIO

import pandas as pd


def tinfo(msg):
    logging.info(f"tid: {threading.current_thread().ident}: " + msg)

def file_path_to_key_path(file_path):
    return file_path.split(os.sep)


def key_to_file_path(key):
    return key


def save_gzip_df_file(file_path, df):
    with open(file_path, "wb") as f:
        with gzip.open(f, "wb", compresslevel=9) as gzf:
            df.to_pickle(gzf)


def read_gzip_df_file(file_path):
    with open(file_path, "rb") as f:
        with gzip.open(f, "rb") as gzf:
            return pd.read_pickle(gzf)


def save_df_file(file_path, df):
    return df.to_pickle(file_path)


def read_df_file(file_path):
    return pd.read_pickle(file_path)


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


def serialize_obj(o):
    bio = BytesIO()
    pickle.dump(o, bio)
    return bio.getvalue()


def deserialize_obj(data):
    bio = BytesIO(data)
    return pickle.load(bio)


def serialize_df(df):
    bio = BytesIO()
    df.to_pickle(bio)
    return bio.getvalue()


def deserialize_df(data):
    bio = BytesIO(data)
    return pd.read_pickle(bio)
