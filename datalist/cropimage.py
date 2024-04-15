import os
import cv2
from pathlib import Path
import numpy as np



def crop_image(origin_path,save_path,cropped_image_height=100,cropped_image_width=100):
    image=cv2.imread(origin_path)
    if image is None: return False
    height=image.shape[0]
    width=image.shape[1]
    cropped_image=image[
        (height//2-cropped_image_height//2):(height//2+cropped_image_height//2),
        (width//2-cropped_image_width//2):(width//2+cropped_image_width//2)
    ]
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)
    cv2.imwrite(save_path,cropped_image)
    return True

def crop_image_full(origin_path,save_path,cropped_image_height=100,cropped_image_width=100):
    image=cv2.imread(origin_path)
    if image is None: return False
    height=376
    width=1241
    cropped_image= np.zeros((376, 1241, 3))
    cropped_image[
        (height//2-cropped_image_height//2):(height//2+cropped_image_height//2),
        (width//2-cropped_image_width//2):(width//2+cropped_image_width//2)
    ]=image
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)
    cv2.imwrite(save_path,cropped_image)
    return True


image_path_3='/content/data/image_03'
image_path_2='/content/drive/MyDrive/Colab Notebooks/data/kitti/2011_10_03_drive_0027_sync/output_CREStereo'
image_local_2='/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo'
depth_path_3='/content/data/2011_10_03_drive_0027_sync/image_03/groundtruth'
depth_path_2='/content/data/2011_10_03_drive_0027_sync/image_02/groundtruth'

# crop_image('/home/ewing/dataset/data_scene_flow/training/image_2/000000_10.png','/home/ewing/dataset/data_scene_flow/training/image_center/000000_10.png')
count_true=0
count_false=0
HEIGHT=200
WIDTH=200
temp=(376, 1241, 3)

# for filename in os.listdir(image_path_2):
#     filepath=image_path_2+'/'+filename
#     image=cv2.imread(filepath)
#     if image is None:
#         print(filepath)
#         continue
#     if image.shape!=temp:
#       print(image.shape)
#     temp=image.shape
#     count_true+=1
#     print(f"\r{count_true}",end='       ')

for filename in os.listdir(image_local_2):
    filepath=image_local_2+'/'+filename
    savepath='/home/ewing/dataset/kitti_test/data/2011_10_03_drive_0027_sync/output_CREStereo_full/'+filename
    flag=crop_image_full(filepath,savepath,HEIGHT,WIDTH)
    if flag==True:
        count_true+=1
    else:
        count_false+=1
    print(f'\rsuccess:{count_true} fail:{count_false}',end='        ')

