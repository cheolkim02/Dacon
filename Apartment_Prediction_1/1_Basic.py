from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

''''''
train = pd.read_csv('train.csv')
train = train.replace({'transaction_date': '1~10'} , 1)
train = train.replace({'transaction_date': '11~20'} , 2)
train = train.replace({'transaction_date': '21~31'} , 3)
train = train.replace({'transaction_date': '21~29'} , 3)
train = train.replace({'transaction_date': '21~30'} , 3)
train = train.replace({'transaction_date': '21~28'} , 3)
train = train.drop(columns=['addr_kr', 'apartment_id'])
''''''

''''''
le_apt = LabelEncoder()
le_apt = le_apt.fit(train['apt'])
train['apt'] = le_apt.transform(train['apt'])

le_city = LabelEncoder()
le_city = le_city.fit(train['city'])
train['city'] = le_city.transform(train['city'])

le_dong = LabelEncoder()
le_dong = le_dong.fit(train['dong'])
train['dong'] = le_dong.transform(train['dong'])

le_jibun = LabelEncoder()
le_jibun = le_jibun.fit(train['jibun'])
train['jibun'] = le_jibun.transform(train['jibun'])

train = train.dropna()
train_x = train.drop(columns=['transaction_id', 'transaction_real_price'])
train_y = train['transaction_real_price']
''''''

''''''
model = LinearRegression()
model = model.fit(train_x, train_y)
''''''

''''''
test = pd.read_csv('test.csv')
test_x = test.drop(columns=['addr_kr', 'apartment_id', 'transaction_id'])
test_x = test_x.replace({'transaction_date': '1~10'} , 1)
test_x = test_x.replace({'transaction_date': '11~20'} , 2)
test_x = test_x.replace({'transaction_date': '21~31'} , 3)
test_x = test_x.replace({'transaction_date': '21~29'} , 3)
test_x = test_x.replace({'transaction_date': '21~30'} , 3)
test_x = test_x.replace({'transaction_date': '21~28'} , 3)
''''''

''''''
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

prediction = model.predict(train_x)

submission = pd.read_csv('submission.csv')
submission['real_price'] = train['transaction_real_price']
submission['prediction'] = prediction
print(submission)