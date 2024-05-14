import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

def load_image_as_array(path):
    """Load an image and convert it to a numpy array."""
    image = Image.open(path).convert('L')  # Convert image to grayscale
    return np.array(image, dtype=np.float32)  # Normalize to [0, 1]

def calculate_mae(image1, image2):
    """Calculate the Mean Absolute Error between two images."""
    mask = image2 > 0.0001  # Create a mask for valid pixels in the second image
    valid_pixels1 = image1[mask]
    valid_pixels2 = image2[mask]
    mae = np.mean(np.abs(valid_pixels1 - valid_pixels2))
    return mae

# Example usage
path1 = r"D:\projects\MyCSPN\output\sgd0512_step24_FT200_normalized_resnet18_edited1024_traindepth_single\eval_result_n\00001_gt.png"
path2 = r"D:\projects\MyCSPN\output\sgd0514_step24_mynet_pretrainedrefine\eval_result1\00001_pred.png"
image1 = load_image_as_array(path1)
image2 = load_image_as_array(path2)
# Calculate MAE
mae = calculate_mae(image1, image2)
print("The MAE loss between the images is:", mae)
