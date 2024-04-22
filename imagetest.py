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
# image = Image.open("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full/0000000005.png")
def imginfo(image):
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
    hist, bins = np.histogram(non_zero_pixels, bins=256, range=(0, np.max(image_data)))

    # 绘制直方图
    plt.figure(figsize=(10, 4))
    plt.bar(bins[:-1], hist, width=bins[1] - bins[0], color='gray', edgecolor='black')
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    # plt.xlim(0, 65535)  # 调整此值以更适合你的具体像素值范围
    plt.show()
# for pixel_value in np.nditer(image_data):
#     print(f"\r{pixel_value}",end='          ')

def changeimage(image_path,save_path=''):
    image=Image.open(image_path).convert('I')
    # img_uint16_cv = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # 将16位图像转换为8位，因为OpenCV的色彩映射通常在8位图像上操作
    # img_uint8_cv = cv2.convertScaleAbs(img_uint16_cv, alpha=(255.0/65535.0))

    # 应用色彩映射
    # color_mapped_img = cv2.applyColorMap(img_uint8_cv, cv2.COLORMAP_MAGMA)
    # cv2.imwrite(save_path, color_mapped_img)
    img_uint16 = np.array(image)  # 转换为NumPy数组
    # imginfo(image)
    img_uint16=np.where(img_uint16 == 0, 0, 2**16-1 - img_uint16)
    img_uint16=((img_uint16/4096)**4).astype(np.uint16)
    img_uint16=Image.fromarray(img_uint16)
    # imginfo(img_uint16)
    
    img_uint16.save(save_path)
    # img_uint8 = (img_uint16//256).astype(np.uint8)  # 右移8位等同于除以256 # 转换为uint8
    # img_uint8=np.where(img_uint8 == 0, 0, 65535 - img_uint8)
    # img_uint8=65535-img_uint8   
    # img_uint8 = Image.fromarray(img_uint8)# 如果需要，转换回PIL图像
    # img_uint8.save(save_path)
    # imginfo(img_uint8)
    
# image = Image.open("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full/0000000005.png")
# image = Image.open("/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/image_02/groundtruth/0000000005.png")
# imginfo(image)
# changeimage("D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\groundtruth\\0000000005.png","D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\groundtruth_uint8\\4.png")
old_path="D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\groundtruth"
new_path="D:\\dataset\\data\\2011_10_03_drive_0027_sync\\image_02\\groundtruth_uint16_4"
for oldfile in os.listdir(old_path):
    changeimage(os.path.join(old_path,oldfile),os.path.join(new_path,oldfile))