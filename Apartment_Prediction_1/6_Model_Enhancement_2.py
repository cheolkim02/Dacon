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
first_model = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=777)
first_model.fit(train_x, train_y)

second_model = RandomForestRegressor(n_estimators=10, max_depth=32, random_state=777)
second_model.fit(train_x, train_y)

# test 데이터 전처리
test = pd.read_csv('test.csv')
test['transaction_year'] = test['transaction_year_month'].astype(str).str[:4].astype(int)
test['transaction_month'] = test['transaction_year_month'].astype(str).str[4:].astype(int)

columns=['transaction_id', 'addr_kr', 'apartment_id', 'transaction_year_month']
test_x = test.drop(columns=columns)

test_x = test_x.replace({'transaction_date': '1~10'} , 1)
test_x = test_x.replace({'transaction_date': '11~20'} , 2)
test_x = test_x.replace({'transaction_date': '21~31'} , 3)
test_x = test_x.replace({'transaction_date': '21~29'} , 3)
test_x = test_x.replace({'transaction_date': '21~30'} , 3)
test_x = test_x.replace({'transaction_date': '21~28'} , 3)

import numpy as np
for city in np.unique(test_x['city']):
    if city not in le_city.classes_:
        le_city.classes_ = np.append(le_city.classes_, city)
test_x['city'] = le_city.transform(test_x['city'])

for dong in np.unique(test_x['dong']):
    if dong not in le_dong.classes_:
        le_dong.classes_ = np.append(le_dong.classes_, dong)
test_x['dong'] = le_dong.transform(test_x['dong'])

for jibun in np.unique(test_x['jibun']):
    if jibun not in le_jibun.classes_:
        le_jibun.classes_ = np.append(le_jibun.classes_, jibun)
test_x['jibun'] = le_jibun.transform(test_x['jibun'])

for apt in np.unique(test_x['apt']):
    if apt not in le_apt.classes_:
        le_apt.classes_ = np.append(le_apt.classes_, apt)
test_x['apt'] = le_apt.transform(test_x['apt'])

# prediction
prediction1 = first_model.predict(test_x)
prediction2 = second_model.predict(test_x)

'''# Ensemble - Blending (average)'''
blended_prediction = prediction1*0.5 + prediction2*0.5

submission = pd.read_csv('sample_submission.csv')
submission['transaction_real_price'] = blended_prediction
submission.to_csv('submission.csv', index=False)