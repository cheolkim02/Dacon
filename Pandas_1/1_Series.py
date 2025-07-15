import pandas as pd
import numpy as np

data = [10, 20, 30, 40, 50]
series1 = pd.Series(data)
print(series1)
print()

np_data = np.array(data)
series2 = pd.Series(np_data)
print(series2)
print()

data_dict = {
    'a': 10,
    'b': 20,
    'c': 30,
    'd': 40,
    'e': 50
}
series = pd.Series(data_dict)
print(series)
print()

data1 = [1, 2, 3, 4]
index = ['a', 'b', 'c', 'd']
series = pd.Series(data1, index=index, name='n', dtype=int)
print(series)
print()


''''''


data = [1,2,3,4,5,6,7,8]
index = ['a','b','c','d','e','f','g','h']
series = pd.Series(data, index=index, name='n', dtype='int32')
np_values = series.values # Series.values => convert to np array
np_index = series.index
np_name = series.name
np_type = series.dtype
series = series.astype('float') # change dtype of pd Series
print(series)
# print(series.head())
print(series.describe()) # statistics


''''''