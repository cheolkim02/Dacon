''' 이미지 불러오기 '''
# PIL(pillow): 이미지 열고 조작하고 저장하는 라이브러리
# plt: 그래프 / 이미지 시각화
from PIL import Image
import matplotlib.pyplot as plt

image_grey_path = 'image_grey.png'
image_grey = Image.open(image_grey_path)




''' 이미지 시각화 '''
import numpy as np
image_grey = np.array(image_grey) # 이미지를 넘파이 어레이로
plt.imshow(image_grey, cmap='gray')
plt.axis('off') 
plt.show()

# 이미지 정보 출력
print(f"이미지 해상도: {image_grey.shape}")
print(f"픽셀 값의 범위: {image_grey.min()} to {image_grey.max()}")

# 이미지를 숫자로 출력
print(image_grey)

# "좀 더 잘 좀 보여줘봐"
''' 흑백 이미지를 dataframe으로 벼환 '''
import pandas as pd

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

df_image_grey = pd.DataFrame(image_grey)
print(df_image_grey)


''' 특정 픽셀 수정 후 시각화 '''
image_grey[5, 5] = 255.0  
image_grey[10, 3] = 255.0
image_grey[25, 4] = 255.0

plt.imshow(image_grey, cmap='gray')
plt.axis('off')
plt.title("Image with Modified Pixels")
plt.show()




''' 이미지 해상도 조정 '''
original_image = Image.open(image_grey_path) # 원래 28 x 28임

# 이미지 해상도 변경
low_resolution_image = original_image.resize((14, 14), Image.Resampling.LANCZOS) # 새 크기, 품질 유지
high_resolution_image = original_image.resize((52, 52), Image.Resampling.LANCZOS) # 새 크기, 품질 유지

# 결과 이미지 시각화
fig, axes = plt.subplots(1, 3, figsize=(12, 4)) # 1행 3열의 서브플롯 생성, (12, 4)는 전체 그림의 크기 설정
# 각 서브플롯(axes[0], [1], [2])에 각각 원본, 낮은 해상도, 높은 해상도 이미지 표시.
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(low_resolution_image, cmap='gray')
axes[1].set_title('Low Resolution (14x14)')
axes[1].axis('off')

axes[2].imshow(high_resolution_image, cmap='gray')
axes[2].set_title('High Resolution (52x52)')
axes[2].axis('off')

plt.show()




''' 컬러 이미지 '''
image_path = 'dog.png'
image_color_open = Image.open(image_path)
image_color = np.array(image_color_open)

plt.imshow(image_color)
plt.axis('off') 
plt.show()

# 이미지 정보 출력
print(f"이미지 해상도: {image_color.shape}")
print(f"픽셀 값의 범위: {image_color.min()} to {image_color.max()}")
print(image_color)


''' 컬러 이미지 RGB 분리 '''
r_channel = image_color[:, :, 0]
g_channel = image_color[:, :, 1]
b_channel = image_color[:, :, 2]

# 원본 컬러 이미지, 흑백 이미지, 그리고 RGB채널 이미지를 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].imshow(image_color)
axes[0, 0].set_title("Original Color Image")
axes[0, 0].axis('off')

axes[0, 1].imshow(r_channel, cmap='Reds')
axes[0, 1].set_title("Red Channel")
axes[0, 1].axis('off')

axes[1, 0].imshow(g_channel, cmap='Greens')
axes[1, 0].set_title("Green Channel")
axes[1, 0].axis('off')

user_check = axes[1, 1].imshow(b_channel, cmap='Blues')
axes[1, 1].set_title("Blues Channel")
axes[1, 1].axis('off')

plt.show()


