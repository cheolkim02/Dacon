import pandas as pd
import numpy as np

data1 = [10, 20, 30, 40]
data2 = [5, 15, 25, 35]
index = ['a', 'b', 'c', 'd']
series1 = pd.Series(data1, index=index)
series2 = pd.Series(data2, index=index)

# 끼리 연산
addition_result = series1 + series2
subtraction_result = series1 - series2
multiplication_result = series1 * series2
division_result = series1 / series2
# print(addition_result)
# print(subtraction_result)
# print(multiplication_result)
# print(division_result)

# 상수 연산
addition_result = series1 + 1
# 지수 연산 (exponential)
squared_series = series1 ** 2
# 로그 연산
log_series = np.log(series1)

'''statistics'''
data = [1, 2, 3, 4, 5, None]
series = pd.Series(data, name='MySeries')

sum_result = series.sum()
mean_result = series.mean()
max_result = series.max()
min_result = series.min()
std_result = series.std()
var_result = series.var()
median_result = series.median()
quantile_result = series.quantile(0.25)  
count_result = series.count()

print(f"Sum: {sum_result}")
print(f"Mean: {mean_result}")
print(f"Max: {max_result}")
print(f"Min: {min_result}")
print(f"Standard Deviation: {std_result}")
print(f"Variance: {var_result}")
print(f"Median: {median_result}")
print(f"25th Percentile: {quantile_result}")
print(f"Count (excluding missing values): {count_result}")

''''''


