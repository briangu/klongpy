import numpy as np
import pandas as pd


def mkdf(x):
    return pd.DataFrame({'col1': np.arange(x)})
