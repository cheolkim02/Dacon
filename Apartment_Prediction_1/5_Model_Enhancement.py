''' build random forest ensemble model. Hyperparameter tuning '''
import pandas as pd
train = pd.read_csv('train.csv')

''' 전처리 시작 '''
''' 1. derived variable '''
train['transaction_year'] = train['transaction_year_month'].astype(str).str[:4].astype(int)
train['transaction_month'] = train['transaction_year_month'].astype(str).str[4:].astype(int)

''' 2. Null imputation '''
null_data = train[train['exclusive_use_area'].isnull()]
median_value = train.groupby('apartment_id')['exclusive_use_area'].median()

null_filled = pd.merge(median_value, null_data, on='apartment_id')
null_filled = null_filled.rename(columns={'exclusive_use_area_x':'exclusive_use_area'})
null_filled = null_filled.drop(columns=['exclusive_use_area_y'])
train_cleaned = train.dropna()
train_filled = pd.concat([train_cleaned, null_filled])

''' 3. select columns for training '''
train_x = train_filled.drop(columns=['transaction_id', 'apartment_id', 'addr_kr', 'transaction_year_month', 'transaction_real_price'])
train_y = train_filled['transaction_real_price']

''' 4. intify - replace '''
train_x = train_x.replace({'transaction_date': '1~10'} , 1)
train_x = train_x.replace({'transaction_date': '11~20'} , 2)
train_x = train_x.replace({'transaction_date': '21~31'} , 3)
train_x = train_x.replace({'transaction_date': '21~29'} , 3)
train_x = train_x.replace({'transaction_date': '21~30'} , 3)
train_x = train_x.replace({'transaction_date': '21~28'} , 3)

''' 5. Label Encoding '''
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
''' 전처리 끝 '''


''' 앙상블 데이터 불러오기: Random Forest ''''''
Bagging: 원본 데이터에서 n개 데이터 추출. 각각으로 decision tree 모델 만들고 (n개), 결과 종합.
random_state=777: random seed. 모델의 시드값을 특정해서 동일한 결과를 얻기 위함.
오로지 데이콘 학습 목적을 위해 랜덤값 동일시키는 거임.
'''
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=777)

''' Hyperparameter Tuning - GridSearchCV ''''''
데이터마다 최적의 하이퍼파라미터는 다르기 때문에 tuning 필요함.
GridSearchCV: 내가 하이퍼파라미터 조합 정해주면, 다 시도해보고 최적을 알려주는 라이브러리
'''
from sklearn.model_selection import GridSearchCV
# 1. param_grid라는 변수에 하이퍼파라미터 조합 담기
# - n_estimators: 랜덤 포레스트 모델을 만들 때 사용할 의사결정나무의 개수.
# - max_depth: 랜덤 포레스트 모델을 만들 때 사용할 의사결정나무의 최대 깊이. 과적합 방지
param_grid = [{
    'n_estimators': [10, 12],
    'max_depth': [30, 32]
    }
]
# 2. grid_search라는 변수에 GridSearchCV를 담기.
# - model: 위에서 만든 random forest 모델.
# - param_grid: (1)에서 만든 하이퍼파라미터 조합을 사용.
# - cv(cross-validation)=2: 받은 데이터를 split0과 split1으로 나눔. split0으로 학습한 모델을 split1으로 평가,
# split1으로 학습한 모델을 split0으로 평가하여 cross-validate.
# - RMSE: 각각(모든 조합)의 모델을 RMSE로 평가. neg: 수가 작으면 좋은거니까 neg 붙혀서 쉽게 성능 비교.
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=2,
    scoring='neg_root_mean_squared_error'
)
# 3. grid_search를 학습 데이터인 train_x와 train_y에 대해 실행해서 최적(fit) 파라미터 찾기.
# - 총 8개 모델 만듦. cv=2이니까 2개의 데이터셋에 대해 가능한 조합=4개 모델 = 총 8개.
grid_search.fit(train_x, train_y)
print('Done')

# 결과 확인
result = pd.DataFrame(grid_search.cv_results_)
# 원하는 컬럼만 골라서 확인
print(result[['params', 'split0_test_score', 'split1_test_score', 'mean_test_score', 'rank_test_score']])
'''
                                  params  split0_test_score  split1_test_score  mean_test_score  rank_test_score
0  {'max_depth': 30, 'n_estimators': 10}      -10020.360862      -16193.822814    -13107.091838                1
1  {'max_depth': 30, 'n_estimators': 12}      -10125.971224      -16237.462231    -13181.716727                4
2  {'max_depth': 32, 'n_estimators': 10}      -10025.724548      -16193.685152    -13109.704850                2
3  {'max_depth': 32, 'n_estimators': 12}      -10129.919592      -16218.791547    -13174.355570                3

rank 1, i.e. neg mean test score가 가장 큰 하이퍼파라미터 조합은 {'max_depth': 30, 'n_estimators': 10}인걸 알 수 있음.
'''

''' 모델 학습 '''
best_model = RandomForestRegressor(n_estimators=10, max_depth=30, random_state=777)
best_model.fit(train_x, train_y)
''''''

''' 테스트 데이터 전처리 '''
test = pd.read_csv('test.csv')
test['transaction_year'] = test['transaction_year_month'].astype(str).str[:4].astype(int)
test['transaction_month'] = test['transaction_year_month'].astype(str).str[4:].astype(int)

test_x = test.drop(columns=['transaction_id', 'apartment_id', 'addr_kr', 'transaction_year_month'])

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

''' prediction '''
prediction = best_model.predict(test_x)
submission = pd.read_csv('submission.csv')
submission['transaction_real_price'] = prediction
submission.to_csv('submission.csv', index=False)



'''
1: 68818.92358720815
2: 6086883.606698486
3: 6233131.13634477

'''