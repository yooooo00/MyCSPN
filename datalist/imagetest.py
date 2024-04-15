import cv2
import os

# image=open('/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth/0000000005.png','r')
# image=cv2.imread('/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth/0000000005.png')
# print(image.shape)
# cv2.imshow('img',image)
# cv2.waitKey()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
# image = Image.open("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth/0000000005.png")
image = Image.open("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full/0000000005.png")

# 输出图像的模式（如 "RGB" 表示红绿蓝三通道，"L" 表示灰度图）
print("Image Mode:", image.mode)

# 转换图像为NumPy数组
image_data = np.array(image)

# 输出数组的数据类型，这表明了每个像素值的存储方式
print("Data Type of Image Array:", image_data.dtype)

# 输出像素值的范围
print("Min Pixel Value:", np.min(image_data))
print("Max Pixel Value:", np.max(image_data))
non_zero_pixels = image_data[image_data > 0]
# hist, bins = np.histogram(image_data, bins=256, range=(0, 256))
hist, bins = np.histogram(non_zero_pixels, bins=256, range=(0, 256))

# 绘制直方图
plt.figure(figsize=(10, 4))
plt.bar(bins[:-1], hist, width=bins[1] - bins[0], color='gray', edgecolor='black')
plt.title('Pixel Value Distribution')
plt.xlabel('Pixel Values')
plt.ylabel('Frequency')
# plt.xlim(0, 65535)  # 调整此值以更适合你的具体像素值范围
plt.show()
for pixel_value in np.nditer(image_data):
    print(f"\r{pixel_value}",end='          ')