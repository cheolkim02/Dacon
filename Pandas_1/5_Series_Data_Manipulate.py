import pandas as pd

data = [1, 2, 3, 4, 5]
series = pd.Series(data)

''' Series.apply() => 시리즈 각 요소에 함수 적용 '''
def custom_function(x) :
    if x>=3 :
        result = x+2
    else :
        result = x+4
    return result

result_series = series.apply(custom_function)
print(result_series)
print()

# lambda: anonymous function. one line function.
result_series = series.apply(lambda x: x+2 if x>=3 else x+4)
print(result_series)
print()



'''mapping'''
# .map() -> only series
# .map() -> maps all elements
# .apply() -> for series and dataframes
# .apply() -> can choose axis to apply.
result_series = series.map(lambda x: x**2)
print(result_series)
print()



'''replace'''
data = ['apple', 'banana', 'cherry']
series = pd.Series(data)

# 1. 'banana'를 'orange'로 replace
result_series = series.replace('banana', 'orange')
print(result_series)

# 2. use a dictionary to replace
replace_dict = {'banana': 'orange', 'apple': 'strawberry'}
result_series = series.replace(replace_dict)
print(result_series)

# 3. use regular expression(regex). 정규표현식.
# 'a'로 시작하는걸 'fruit'으로 replace
result_series = series.replace('^a', 'fruit', regex=True)
print(result_series)
# 번외: 모든 'a'를 'fruit'으로 replace
result_series = series.replace('a', 'fruit', regex=True)
print(result_series)