import numpy as np

def normalize_image(image, scale=255):
    """ Normalize the image to [0, scale] range. """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    scaled_image = (normalized_image * scale).astype(np.uint8)  # Scale to [0, scale]
    return scaled_image

# Example usage
import cv2
def normalize_and_write(old_path,new_path):
    gray_image = cv2.imread(old_path, cv2.IMREAD_GRAYSCALE)
    normalized_gray_image = normalize_image(gray_image)
    # cv2.imshow("Normalized Gray Image", normalized_gray_image)
    # cv2.waitKey(0)
    cv2.imwrite(new_path,normalized_gray_image)

old_dir = r"D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse_4_uint8"
new_dir = r"D:\dataset\data\FallingThings\kitchen_0_result\left_gt_reverse_4_uint8_normalized"

import os
for filename in os.listdir(old_dir):
    old_path = os.path.join(old_dir, filename)
    new_path = os.path.join(new_dir, filename)
    normalize_and_write(old_path,new_path)