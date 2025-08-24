from datetime import datetime
from datetime import timedelta
from datetime import date
from datetime import time
# timedelta: calculate date differences


''' datetime 객체 생성하고 연산하기 '''
now = datetime.now()
specific_date = datetime(2002, 2, 8, 14, 11)

n_days = 100
future_date = now + timedelta(days=n_days)
print(future_date)
print(future_date.date) # 시간 없이 날짜만 저장

time_diff = future_date - now
print(time_diff)
print(type(time_diff))

# 날짜 차이를 시간으로 보려면 '초'로 먼저 계산 후 3600으로 나눠주면 됨.
hours_diff = time_diff.total_seconds()/3600

yesterday = now - timedelta(days=1)




''' date 클래스 생성과 날짜 비교 '''
today = datetime.today()
special_day = datetime(2024, 12, 25)

if today < special_day:
    print("특별한 날이 아직 오지 않았습니다.")
elif today == special_day:
    print("오늘은 특별한 날입니다!")
else:
    print("특별한 날이 지났습니다.")


''' time 클래스 생성과 날짜 비교 '''
now = datetime.now()
current_date = now.date() # 날짜만
current_time = now.time() # 시간만

start_time = time(9, 0)
end_time = time(17, 0)
if start_time <= current_time <= end_time:
    print("현재는 업무 시간입니다.")
else:
    print("현재는 업무 시간이 아닙니다.")


''' combine 'date' and 'time' to make 'datetime' '''
meeting_date = date(2024, 1, 15)
meeting_time = time(14, 30)
meeting_datetime = datetime.combine(meeting_date, meeting_time)



''' formatting: 'strftime' '''
now = datetime.now()

basic_format = now.strftime('%Y-%m-%d %H:%M:%S') # (예시: 2024-08-07 05:40:28)
korean_format = now.strftime('%Y년 %m월 %d일 %H시 %M분 %S초') # (예시: 2024년 08월 07일 05시 40분 28초)
date_only_format = now.strftime('%Y/%m/%d') # (예시: 2024/08/07)
time_12h_format = now.strftime('%I:%M %p') # (예시: 05:40 AM)
weekday_month_format = now.strftime('%A, %d %B %Y') # (예시: Wednesday, 07 August 2024)
iso_format_basic = now.strftime('%Y-%m-%dT%H:%M:%S') # (예시: 2024-08-07T05:40:28)
iso_format_with_timezone = now.strftime('%Y-%m-%dT%H:%M:%S%z') # (예시: 2024-08-07T05:40:28+0900)

