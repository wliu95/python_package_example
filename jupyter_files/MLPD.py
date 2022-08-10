

import pandas as pd

import warnings
warnings.filterwarnings("ignore")
import collections
from datetime import datetime
def rank_d(data,varia,buckets):

    rank = varia + '_rank'
    data[rank] = 0
    data.sort_values(by = varia, inplace = True)
    data.reset_index(drop = True,inplace = True)
    seperate = len(data) // buckets
    for i in data.index:
        data[rank][i] = min(i // seperate + 1,buckets)
    return data


