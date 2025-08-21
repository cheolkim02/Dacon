import numpy as np
import matplotlib.pyplot as plt

''''''''''''''''''''''''' BLACK AND WHITE IMAGE '''''''''''''''''''''''''
''' example image and filter '''
image_grey = np.array([
    [1, 1, 0],
    [0, 1, 0],
    [1, 0, 1]
])

filter_grey = np.array ([
    [1, 0],
    [0, 1]
])
''''''

''' create feature map by applying convolution'''
def apply_convolution(image_grey, filter_grey) :
    # feature map의 크기 설정
    size = filter_grey.shape[0]
    height, width = image_grey.shape
    result = np.zeros((height - size + 1, width - size + 1)) # 외우면 됨
    
    for i in range(result.shape[0]) :
        for j in range(result.shape[1]) :
            result[i, j] = np.sum(image_grey[i:i+size, j:j+size] * filter_grey)
    return result # returning the np array of the finished feature map

feature_map = apply_convolution(image_grey, filter_grey)

''' visualize: original image, feature map '''
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image_grey')
plt.imshow(image_grey, cmap='gray', interpolation='nearest')

plt.subplot(1, 2, 2)
plt.title('Feature Map')
plt.imshow(feature_map, cmap='gray', interpolation='nearest')

plt.show()
''''''



'''''''''''''''''''''''''''' COLOR IMAGE '''''''''''''''''''''''''''''
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
print("필터 값:\n",filter_color) 
print("필터의 형태:\n",filter_color.shape)

def apply_3d_convolution(image_color, filter_color) :
    filter_size_x, filter_size_y, filter_depth = filter_color.shape
    image_color_height, image_color_width, image_color_depth = image_color.shape
    output_height = image_color_height - filter_size_x + 1 # +1 b/c 'stride' is 1.
    output_width = image_color_width - filter_size_y + 1
    output = np.zeros((output_height, output_width))
    for x in range(output_height) :
        for y in range(output_width) :
            output[x, y] = np.sum(image_color[x:x+filter_size_x, y:y+filter_size_y, :] * filter_color)
    return output

feature_map = apply_3d_convolution(image_color, filter_color)
# print(feature_map)