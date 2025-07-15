import pandas as pd

data = [20, 10, 40, 50, 60, 30, 70, 80]
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
series = pd.Series(data, index=index, name='my_series')

series_sorted = series.sort_values()
series_inverse = series.sort_values(ascending=False)

series_sorted = series.sort_index()
series_inverse = series.sort_index(ascending=False)

series_rank = series.rank()
series_rank = series.rank(ascending=False)
print(series_rank)