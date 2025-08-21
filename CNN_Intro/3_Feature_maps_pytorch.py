import numpy as np
import torch
import torch.nn as nn

red_channel = np.array([
    [1, 2, 0, 2, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 2, 0, 1],
    [0, 1, 1, 2, 0],
    [1, 0, 1, 0, 0]
])
green_channel = np.array([
    [0, 1, 1, 0, 1],
    [0, 2, 1, 1, 0],
    [0, 0, 2, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 0, 2, 2, 0]
])
blue_channel = np.array([
    [1, 0, 2, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 0, 2, 1, 2],
    [1, 0, 1, 0, 0],
    [0, 0, 1, 2, 0]
])
image_color = np.stack((red_channel, green_channel, blue_channel), axis=-1)
# np.stack with axis=-1 을 하면 위아래로 쌓임. 쌓여서 만들어진 벡터 (ex. (1, 0, 1))은 각 픽셀의 RGB값.

# make filters - one per layer, then stack them using np.stack with axis=-1
filter_red_channel = np.array([[1, 0], [0, 1]])
filter_green_channel = np.array([[1, 2], [0, 1]])
filter_blue_channel = np.array([[2, 0], [0, 0]])
filter_color = np.stack((filter_red_channel, filter_green_channel, filter_blue_channel), axis=-1)
'''
# 컨볼루션을 위한 이미지와 필터 텐서 형태 조정
# unsqueeze(0): 첫번째 위치에 차원 추가. = 배치(batch) 크기 나타냄.
# 국룰 pytorch 신경망 모델 입력 형태: 배치 크기, 채널 수, 높이, 너비 (배치 크기 = 한번에 처리할 수 있는 이미지 수)
# permute:
# 0 -> 첫 번째 차원은 그대로 (배치 크기)
# 3 -> 원래 네 번째 차원(원래 데이터의 채널)을 두 번째 위치로 이동
# 1 -> 원래 두 번째 차원(높이)을 세 번째 위치로 이동
# 2 -> 원래 세 번째 차원(너비)을 네 번째 위치로 이동
# 국룰 입력 형태에 맞추는 거임.
'''
input_tensor = torch.tensor(image_color, dtype=torch.float).unsqueeze(0)
print("이미지 텐서에 차원 추가 :\n", input_tensor.shape)
input_tensor = input_tensor.permute(0, 3, 1, 2)
print("이미니 텐서의 차원을 재배열 :\n", input_tensor.shape)
filter_tensor = torch.tensor(filter_color, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)
print("필터 텐서의 차원 추가 + 재배열 :\n",filter_tensor.shape)

''' 컨볼루션 레이어의 정의와 파라미터 설정 '''
# nn.Conv2d -> 2차원 컨볼루션 레이어 생성. 이 레이어는 입력 데이터에 2D 컨볼루션 연산 적용
# in_channels=3 -> r, g, b 각각 있으니까 3.
# out_channels-1 -> 컨볼루션 레이어 적용 후 생성되는 특징 맵(feature map) 수. (내가 정함)
# kernel_size=2 -> 컨볼루션 필터가 2x2임을 의미. 
conv_layer = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2)
with torch.no_grad() :
    conv_layer.weight = nn.Parameter(filter_tensor)
    conv_layer.bias = nn.Parameter(torch.zeros(1))
# no_grad() -> 그라디언트 계산 비활성화(파라미터 업데이트되지 않도록). 무델 가중치/바이어스 초기화/설정할때 사용
# weight -> 레이어의 필터 가중치 -> filter_tensor를 레이어의 가중치로 설정함.
# nn.Parameter -> 텐서를 모듈의 파라미터로 등록하여 자동으로 그라디언트 계산할 수 있게 함.
# 여기선 그라디언트 계산이 필요 없으므로 torch.no_grad() 내부에서 사용해야됨.
# bias -> 바이어스 0으로 초기화. 바이어스가 1개여서 (1).


''' 컨볼루션 레이어 적용 및 결과 (feature map) 출력 '''
output_tensor = conv_layer(input_tensor)
print('Output shape :\n', output_tensor.shape)
print("Output tensor :\n", output_tensor)