from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

''''''
# 데이터 불러오고 결측 데이터 확인
train = pd.read_csv('train.csv')
print(train.info)
''''''

''''''
# exclusive_use_area 컬럼에 결측 데이터 존재.
# 해당 컬럼의 값이 null인 데이터(row)만 선택하기(isnull()).
null_data = train[train['exclusive_use_area'].isnull()]
''''''

''''''
# 컬럼 간 관계 탐색: 도수 분포
# 같은 아파트에서 exclusive_use_area가 어떻게 나타나는지 도수 분포로 확인
import matplotlib.pyplot as plt

plt.figure(figsize = (6,6))

x = train[train['apartment_id'] == 1878]['exclusive_use_area']
plt.hist(x, label='histogram')

plt.xlabel('exclusive_use_area')
plt.ylabel('number of apartment')

plt.legend()
plt.show()
''''''

''''''
# 결측값 대치 - median으로 대치하기
# 1. median 찾기 - 각 아파트 고유 아이디별로 미디언 찾음
median_value = train.groupby('apartment_id')['exclusive_use_area'].median()
print(median_value)
# 2. merge하기
null_filled = pd.merge(median_value, null_data, on='apartment_id')
# 3. 형태 통일시키기
# merge과정에서 exclusive_use_area컬럼 이름이 중복돼서
# exclusive_use_area_x와 exclusive_use_area_y로 이름이 변경됐어.
# null값을 overwrite하는 게 아니라 그냥 칼럼은 하나 추가한 것 뿐이기 때문이지.
# exclusive_use_area_x를 원래 이름으로 변경하고 나머지 칼럼은 제거하자.
null_filled = null_filled.rename(columns={'exclusive_use_area_x':'exclusive_use_area'})
null_filled = null_filled.drop(columns=['exclusive_use_area_y'])
# 4. 결측값 대치 완료된 데이터 만들기
# train 데이터에서 결측값을 가진 data(row) 없앤 후, null_filled와 합치기
train_cleaned = train.dropna()
train_filled = pd.concat([train_cleaned, null_filled])
''''''

'''전처리'''
# drop
train_x = train_filled.drop(columns=['transaction_real_price', 'transaction_id', 'apartment_id', 'addr_kr'])
train_y = train_filled['transaction_real_price']

# replace
train_x = train_x.replace({'transaction_date': '1~10'} , 1)
train_x = train_x.replace({'transaction_date': '11~20'} , 2)
train_x = train_x.replace({'transaction_date': '21~31'} , 3)
train_x = train_x.replace({'transaction_date': '21~29'} , 3)
train_x = train_x.replace({'transaction_date': '21~30'} , 3)
train_x = train_x.replace({'transaction_date': '21~28'} , 3)

# Label Encoding
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
''''''

''' Train Model '''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
reg = LinearRegression().fit(train_x, train_y)
''''''

''' Test Data Manipulation'''
# read
test = pd.read_csv('test.csv')
# drop
test_x = test.drop(columns=['transaction_id', 'addr_kr', 'apartment_id'])
# replace
test_x = test_x.replace({'transaction_date': '1~10'} , 1)
test_x = test_x.replace({'transaction_date': '11~20'} , 2)
test_x = test_x.replace({'transaction_date': '21~31'} , 3)
test_x = test_x.replace({'transaction_date': '21~29'} , 3)
test_x = test_x.replace({'transaction_date': '21~30'} , 3)
test_x = test_x.replace({'transaction_date': '21~28'} , 3)

# Label Encoding
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
''''''

''' Make Prediction '''
prediction = reg.predict(test_x)
''''''

''' Submission '''
submission = pd.read_csv('sample_submission.csv')
submission['transaction_real_price'] = prediction
submission.to_csv('submission.csv', index=False)