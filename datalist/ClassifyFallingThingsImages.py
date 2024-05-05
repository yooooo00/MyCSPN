import os
import shutil


origin_dir="D:\\dataset\\data\\FallingThings\\kitchen_0"
new_dir_left="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\left"
new_dir_right="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\right"

# for filename in os.listdir(origin_dir):
#     if filename.split('.')[-1]=='jpg' and filename.split('.')[-2]=='left':
#         shutil.copy(os.path.join(origin_dir,filename),os.path.join(new_dir_left,filename))

# for filename in os.listdir(origin_dir):
#     if filename.split('.')[-1]=='jpg' and filename.split('.')[-2]=='right':
#         shutil.copy(os.path.join(origin_dir,filename),os.path.join(new_dir_right,filename))
import cv2
from pathlib import Path
import numpy as np


def crop_image(origin_path,save_path,cropped_image_height=200,cropped_image_width=200):
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

center_dir_left="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\center\\left_200"
center_dir_right="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\center\\right_200"
# for filename in os.listdir(new_dir_left):
#     crop_image(os.path.join(new_dir_left,filename),os.path.join(center_dir_left,filename))
# for filename in os.listdir(new_dir_right):
#     crop_image(os.path.join(new_dir_right,filename),os.path.join(center_dir_right,filename))

cre_dir="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\output_CREStereo_200"
cre_full_dir="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\output_CREStereo_full_200"
cre_gray_dir="D:\\dataset\\data\\FallingThings\\kitchen_0_result\\output_CREStereo_gray"

from PIL import Image
# for filename in os.listdir(cre_dir):
#     image=Image.open(os.path.join(cre_dir,filename)).convert('L')
#     image.save(os.path.join(cre_gray_dir,filename))

def crop_image_full(origin_path,save_path,cropped_image_height=200,cropped_image_width=200):
    # image=cv2.imread(origin_path)
    image=Image.open(origin_path).convert('L')
    if image is None: 
        print(origin_path)
        return False
    height=540
    width=960
    cropped_image= np.zeros((540, 960))
    cropped_image[
        (height//2-cropped_image_height//2):(height//2+cropped_image_height//2),
        (width//2-cropped_image_width//2):(width//2+cropped_image_width//2)
    ]=image
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent)
    # cv2.imwrite(save_path,cropped_image)
    cropped_image=Image.fromarray(cropped_image).convert('L')
    cropped_image.save(save_path)
    return True

for filename in os.listdir(cre_dir):
    crop_image_full(os.path.join(cre_dir,filename),os.path.join(cre_full_dir,filename))

gt_dir='D:\\dataset\\data\\FallingThings\\kitchen_0_result\\left_gt'

# for filename in os.listdir(origin_dir):
#     if filename.split('.')[-1]=='png' and filename.split('.')[-2]=='depth' and filename.split('.')[-3]=='left':
#         shutil.copy(os.path.join(origin_dir,filename),os.path.join(gt_dir,filename))

gt_reverse_dir=r"D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse"

# for filename in os.listdir(gt_dir):
#     image=Image.open(os.path.join(gt_dir,filename)).convert('I')
#     image=np.array(image)
#     image=2**16-1-image
#     image=Image.fromarray(image)
#     image.save(os.path.join(gt_reverse_dir,filename))

gt_4_dir=r'D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse_4'

# for filename in os.listdir(gt_reverse_dir):
#     image=Image.open(os.path.join(gt_reverse_dir,filename)).convert('I')
#     image=np.array(image)
#     image=(image/2**12)**4
#     image=Image.fromarray(image).convert('I')
#     image.save(os.path.join(gt_4_dir,filename))

gt_4_uint8_dir=r'D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse_4_uint8'

# for filename in os.listdir(gt_4_dir):
#     image=Image.open(os.path.join(gt_4_dir,filename))
#     image=np.array(image)
#     image=image//2**8
#     image=Image.fromarray(image).convert('L')
#     image.save(os.path.join(gt_4_uint8_dir,filename))

left_depth_dir=r'D:\dataset\data\FallingThings\kitchen_0_result\left_depth'
left_depth_uint8_dir=r'D:\dataset\data\FallingThings\kitchen_0_result\left_depth_uint8'

# for filename in os.listdir(left_depth_dir):
#     image=Image.open(os.path.join(left_depth_dir,filename)).convert('L')
#     # image=np.array(image)
#     # image=image//2**8
#     # image=Image.fromarray(image).convert('L')
#     image.save(os.path.join(left_depth_uint8_dir,filename))