import pandas as pd

# make series
data = [20, 10, 40, 50, 60, 30, 70, 80]
index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
series = pd.Series(data, index=index, name='my_series')

# indexing
value_at_b = series['b']
value_at_b = series.loc['b']
value_at_three = series.iloc[3]

# slicing
values_b_to_d = series.loc['b':'d']
print(values_b_to_d)
values_2_to_before_5 = series.iloc[2:5]
print(values_2_to_before_5)
print()

# filtering - conditions
condition = series > 30
filtered_series = series[condition]
print(filtered_series)

