from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

train = pd.read_csv('train.csv')
# print(train.info)

null_data = train[train['exclusive_use_area'].isnull()]
# print(null_data)