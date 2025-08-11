import pandas as pd

''' read file '''
train = pd.read_csv('train.csv')

''' make validation dataset - split from train'''
# import from sklearn
from sklearn.model_selection import train_test_split
# train data 70%, validation data 30%
train_split, validation = train_test_split(train, train_size=0.7)

''' 전처리 '''
# 이번에는 결측값 fill없이 그냥 drop할게요
train_split = train_split.dropna()
validation = validation.dropna()

# 사용할 x, y 분리 - use 5 columns for x axis for simplicity
train_x = train_split[['exclusive_use_area', 'year_of_completion', 'transaction_year_month', 'floor', 'apartment_id']]
train_y = train_split['transaction_real_price']
valid_x = validation[['exclusive_use_area', 'year_of_completion', 'transaction_year_month', 'floor', 'apartment_id']]
valid_y = validation['transaction_real_price']

'''
저희가 현재 사용하는 데이터는 총 5개의 컬럼(column)으로 이루어진 x와
1개의 컬럼으로 이루어진 y입니다.

이는 고려해야 하는 변수가 6개라는 뜻으로 6차원의 데이터임을 뜻합니다.
6차원의 데이터는 시각화하기 어렵기 때문에,
표현이 쉬운 2차원의 데이터로 바꿔줘야 할 필요가 있습니다!

이렇게 고차원의 데이터를 저차원의 데이터로 변경해주는 것을 차원축소라고 하고,
저희는 이런 차원축소 기법 중 PCA를 사용할 겁니다.

그런데 PCA를 사용할 때는 각 컬럼의 단위가 중요하기 때문에,
저희는 sklearn에서 제공하는 StandardScaler를 이용해서
데이터를 표준화(standardize) 하겠습니다!
컬럼간 단위가 조정되지 않은 데이터로 PCA하면, 단위가 크거나 분산이 큰 일부 컬럼이
주성분분석에 큰 영향을 미칠 수 있음 = 전체 데이터를 잘 설명하지 못함

아래 코드를 StandardScaler를 만들어 보세요!

이 코드는 먼저 sklearn.preprocessing에서 StandardScaler를 불러온 후,
scaler이름으로 객체를 생성하는 코드입니다.
'''
# Standard Scaling. standardize: 표준 0, 표준편차 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_x) # 주어진 데이터의 각 컬럼별 평균과 표준편차 기억.

train_x_scaled = scaler.transform(train_x) # train_x 표준화하여 train_x_scaled에 담기
valid_x_scaled = scaler.transform(valid_x) # valid_x 표준화하여 valid_x_scaled에 담기
# valid_x에 대해서 fit()을 하지 않는 이유는, validation 데이터를 전혀 모른다고 가정해야 하기 때문

# PCA(주성분분석) - Principan Component Analysis - 주성분을 골라서 차원축소
# 축소시키기 원하는 차원의 크기(주성분의 수)를 지정할 수 있음. 1로 해볼게요.
# x축 5개 컬럼을 1개로 축소하여 간단하게 시각화할 수 있습니다.
# PCA 변환기 만들기
from sklearn.decomposition import PCA
pca = PCA(n_components=1)

# 변환기로 차원축소 진행
pca.fit(train_x_scaled)
train_x_pca = pca.transform(train_x_scaled)
valid_x_pca = pca.transform(valid_x_scaled)

''' Linear Regression 모델 평가 '''
# 1. 모델 학습
from sklearn.linear_model import LinearRegression
model_lr = LinearRegression().fit(train_x_pca, train_y)

# 2. 학습 현황 확인
import matplotlib.pyplot as plt
plt.figure(figsize = (12,6))
# 학습 데이터 시각화
plt.scatter(train_x_pca, train_y, label='train data') #1
plt.xlabel('train_x')
plt.ylabel('train_y')
# 모델의 학습 시각화
x_sorted = train_x_pca.tolist() #2
x_sorted.sort() #3
pred_lr = model_lr.predict(x_sorted) #4
plt.plot(x_sorted, pred_lr, color='red', label='linear regression') #5
plt.legend()
plt.show()

# 3. 검증 데이터로 평가
plt.figure(figsize = (12,6))

plt.scatter(valid_x_pca, valid_y, label='validation data')
plt.xlabel('valid_x')
plt.ylabel('valid_y')

x_sorted = valid_x_pca.tolist() 
x_sorted.sort()
pred_lr = model_lr.predict(x_sorted) 
plt.plot(x_sorted, pred_lr, color='red', label='linear regression') 

plt.legend()
plt.show()


''' 4. Decision Tree 모델 평가 '''
# 1. 모델 학습
from sklearn.tree import DecisionTreeRegressor
model_dt = DecisionTreeRegressor()
model_dt.fit(train_x_pca, train_y)

# 2. 모델의 학습 시각화
plt.figure(figsize = (12,6))

plt.scatter(train_x_pca, train_y, label='train data')
plt.xlabel('train_x')
plt.ylabel('train_y')

x_sorted = train_x_pca.tolist() 
x_sorted.sort()
pred_dt = model_dt.predict(x_sorted) 
plt.plot(x_sorted, pred_dt, color='red', label='decision tree') 

plt.legend()
plt.show()

# 3. 검증 데이터로 평가
plt.figure(figsize = (12,6))

plt.scatter(valid_x_pca, valid_y, label='validation data')
plt.xlabel('valid_x')
plt.ylabel('valid_y')

x_sorted = valid_x_pca.tolist() 
x_sorted.sort()
pred_dt = model_dt.predict(x_sorted) 
plt.plot(x_sorted, pred_dt, color='red', label='decision tree') 

plt.legend()
plt.show()