import pandas as pd

train = pd.read_csv('train.csv')

train['transaction_year'] = train['transaction_year_month'].astype(str).str[:4].astype(int)
train['transaction_month'] = train['transaction_year_month'].astype(str).str[4:].astype(int)

null_data = train[train['exclusive_use_area'].isnull()]
median_value = train.groupby('apartment_id')['exclusive_use_area'].median()

null_filled = pd.merge(median_value, null_data, on='apartment_id')
null_filled = null_filled.rename(columns={'exclusive_use_area_x':'exclusive_use_area'})
null_filled = null_filled.drop(columns=['exclusive_use_area_y'])
train_cleaned = train.dropna()
train_filled = pd.concat([train_cleaned, null_filled])

train_x = train_filled.drop(columns=['transaction_id', 'apartment_id', 'addr_kr', 'transaction_year_month', 'transaction_real_price'])
train_y = train_filled['transaction_real_price']

train_x = train_x.replace({'transaction_date': '1~10'} , 1)
train_x = train_x.replace({'transaction_date': '11~20'} , 2)
train_x = train_x.replace({'transaction_date': '21~31'} , 3)
train_x = train_x.replace({'transaction_date': '21~29'} , 3)
train_x = train_x.replace({'transaction_date': '21~30'} , 3)
train_x = train_x.replace({'transaction_date': '21~28'} , 3)

from sklearn.preprocessing import LabelEncoder
le_city = LabelEncoder()
le_city = le_city.fit(train_x['city'])
train_x['city'] = le_city.transform(train_x['city'])

le_dong = LabelEncoder()
le_dong = le_dong.fit(train_x['dong'])
train_x['dong'] = le_dong.transform(train_x['dong'])

le_jibun = LabelEncoder()
le_jibun = le_jibun.fit(train_x['jibun'])
train_x['jibun'] = le_jibun.transform(train_x['jibun'])

le_apt = LabelEncoder()
le_apt = le_apt.fit(train_x['apt'])
train_x['apt'] = le_apt.transform(train_x['apt'])


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=777)

from sklearn.model_selection import GridSearchCV
param_grid = [{
    'n_estimators': [10, 12],
    'max_depth': [30, 32]
    }
]
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=2,
    scoring='neg_root_mean_squared_error'
)
grid_search.fit(train_x, train_y)
print('Done')

result = pd.DataFrame(grid_search.cv_results_)
print(result[['params', 'split0_test_score', 'split1_test_score', 'mean_test_score', 'rank_test_score']])


'''
result 결과를 보면
rank 1: n_estimators = 10, max_depth = 30.
rank 2: n_estimators = 10, max_depth = 32.

2가 1보다 max_depth가 더 높기에, 더 복잡한 관계도 표현 가능.
조합 하나만 고르지 말고, 두 개 다 쓰자!

저희는 이 두 모델을 사용해서 앙상블(Ensemble)을 진행하겠습니다 == Blending 기법
'''
