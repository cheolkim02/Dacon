import pandas as pd
import datetime
from datetime import date, time, timedelta
import matplotlib.pyplot as plt

''' timestamp '''
ts1 = pd.Timestamp('2023-07-29 12:30:00') # create timestamp - pandas
date_string = '2023-07-29'
ts2 = pd.Timestamp(date_string)


''' change time '''
ts = pd.Timestamp('2023-07-29')
ts_year = ts.year
ts_month = ts.month
ts_day = ts.day
ts_dayofweek = ts.dayofweek

# print(ts)
ts_utc = ts.tz_localize('UTC')
# print(ts_utc)
ts_kst = ts_utc.tz_convert('Asia/Seoul')
# print(ts_kst)


''' convert dataframe string column to timestamp '''
apple = pd.read_csv('AAPL.csv')
print(apple['Date'].dtype) # 변환 전 컬럼 타입
print(type(apple['Date'][0])) # 변환 전 데이터 타입
apple['Date'] = pd.to_datetime(apple['Date'])
print(apple['Date'].dtype) # 변환 후 컬럼 타입
print(type(apple['Date'][0])) # 변환 후 데이터 타입

''' 시간대 (timezone) 변환 '''
apple_data = apple.copy()
apple_data['Date_EST'] = apple_data['Date'].dt.tz_localize('America/New_York')
apple_data['Date_KST'] = apple_data['Date_EST'].dt.tz_convert('Asia/Seoul')
print(apple_data[['Date', 'Date_EST', 'Date_KST']].head())


''' 특정 시간대 데이터 추출 '''
apple_data = apple.copy()
two_thousands = apple_data[
    (apple_data['Date'].dt.date >= pd.to_datetime('2000-01-01').date()) &
    (apple_data['Date'].dt.date <= pd.to_datetime('2009-12-21').date())
]
print(two_thousands.head())

# 시간이 있는 데이터면 가능. 여긴 날짜만 있어서 안됨.
# apple_data = apple.copy()
# apple_data = apple_data.set_index('Date')
# two_thousands2 = apple_data.between_time('09:30', '16:00')
# print(two_thousands2.head())


''' Resampling - 시간 단위를 조정해 데이터를 축소하거나 확대하기'''
# 비슷한듯 다른 효과.
apple_data = apple.copy()
apple_data = apple_data.set_index('Date')
apple_monthly1 = apple_data.resample('1m').last()
print(apple_monthly1.head())

print("으악!")

apple_data = apple.copy()
apple_monthly2 = apple_data.resample('1m', on='Date').last()
print(apple_monthly2.head())



''' 2000년대 거래량 시각화 '''
apple_data = apple.copy()
apple_data['Date'] = pd.to_datetime(apple_data['Date'])
apple_data = apple_data[
    (apple_data['Date'].dt.date >= pd.to_datetime('2000-01-01').date()) &
    (apple_data['Date'].dt.date <= pd.to_datetime('2009-12-21').date())
]
apple_data = apple_data.set_index('Date')

# 월별 거래량 계산 (거래량의 합계)
monthly_volume = apple_data['Volume'].resample('1m').last()

# 월별 거래량을 막대 그래프로 시각화
plt.figure(figsize=(10, 4))
plt.bar(monthly_volume.index, monthly_volume, color='skyblue')
# 그래프 제목 및 레이블
plt.title('AAPL Daily Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.show()


''' 날짜 범위 생성 (날짜 정보만 남기며) 및 날짜 차집합 구하기 '''
# 날짜 범위 생성
date_range = pd.date_range(start='2000-01-01', periods=10, freq='Y')
print("Original Date Range with Time:")
print(date_range)

# normalize() 함수로 시간 정보 제거 (날짜만 남기는 건데, 난 원래 날짜만 있었어서 동일함.)
normalized_dates = date_range.normalize()
print("\nNormalized Date Range (Date Only):")
print(normalized_dates)

# 차집합 계산
additional_dates = pd.date_range(start='2005-01-01', periods=5, freq='Y')
print("\nAdditional Dates:")
print(additional_dates)

# 차집합 계산하여 차이점 출력
difference_dates = normalized_dates.difference(additional_dates)
print("\nDifference (Dates not in Additional Dates):")
print(difference_dates)



''' 공휴일 추정하기 '''
apple_data = apple.copy()
apple_data['Date'] = pd.to_datetime(apple_data['Date'])
apple_data = apple_data.set_index('Date')

# 거래가 이루어진 날짜 추출 (normalize 사용)
trading_days = apple_data.index.normalize().unique()

# 분석 기간의 모든 평일(월-금) 날짜 생성
weekdays = pd.date_range(start=apple_data.index.min().normalize(), 
                         end=apple_data.index.max().normalize(), freq='B')

# 평일에서 거래된 날짜를 제거한 후 남은 날짜를 구하기 (거래가 이루어지지 않은 날짜)
non_trading_days = weekdays.difference(trading_days)

print("공휴일일 가능성이 있는 날짜들:\n", non_trading_days)