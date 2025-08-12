import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

''' read file '''
train = pd.read_csv('train.csv')

''' create derived variable '''
train['transaction_year'] = train['transaction_year_month'].astype(str).str[:4].astype(int)
train['transaction_month'] = train['transaction_year_month'].astype(str).str[4:].astype(int)

''' histogram - transaction_month '''
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.hist(train['transaction_month'], bins=24, label='monthly transaction count') #bins 값 24
plt.xlabel('transaction_month')
plt.ylabel('transaction count')
plt.legend()
plt.show()

''' histogram - transaction_year '''
plt.figure(figsize=(6, 6))
plt.hist(train['transaction_year'], bins=24, label='yearly transaction count')
plt.xlabel('transaction_year')
plt.ylabel('transaction count')
plt.legend()
plt.show()

''' null값 대치 - 'exclusive_use_area' '''
null_data = train[train['exclusive_use_area'].isnull()]
median_value = train.groupby('apartment_id')['exclusive_use_area'].median()

null_filled = pd.merge(median_value, null_data, on='apartment_id')
null_filled = null_filled.rename(columns={'exclusive_use_area_x':'exclusive_use_area'})
null_filled = null_filled.drop(columns=['exclusive_use_area_y'])

train_cleaned = train.dropna()
train_filled = pd.concat([train_cleaned, null_filled])

''' 모델 학습 전 train 전처리 '''
train_x = train_filled.drop(columns=['transaction_id', 'apartment_id', 'addr_kr', 'transaction_year_month', 'transaction_real_price'])
train_y = train_filled['transaction_real_price']

train_x = train_x.replace({'transaction_date': '1~10'} , 1)
train_x = train_x.replace({'transaction_date': '11~20'} , 2)
train_x = train_x.replace({'transaction_date': '21~31'} , 3)
train_x = train_x.replace({'transaction_date': '21~29'} , 3)
train_x = train_x.replace({'transaction_date': '21~30'} , 3)
train_x = train_x.replace({'transaction_date': '21~28'} , 3)

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

''' 모델 훈련 '''
model = LinearRegression()
reg = LinearRegression().fit(train_x, train_y)

''' 예측 전 test 전처리 - 파생변수 생성 '''
test = pd.read_csv('test.csv')
test['transaction_year'] = test['transaction_year_month'].astype(str).str[:4].astype(int)
test['transaction_month'] = test['transaction_year_month'].astype(str).str[4:].astype(int)

test_x = test.drop(columns=['transaction_id', 'apartment_id', 'addr_kr', 'transaction_year_month', 'transaction_real_price'])

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

''' 예측 '''
prediction = reg.predict(test_x)
submission = pd.read_csv('submission.csv')
submission['transaction_real_price'] = prediction
submission.to_csv('submission.csv', index=False)
