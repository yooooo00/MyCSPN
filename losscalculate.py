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

def calculate_mse(image1, image2):
    """Calculate the Mean Squared Error between two images."""
    mask = image2 > 0.0001  # Create a mask for valid pixels in the second image
    valid_pixels1 = image1[mask]
    valid_pixels2 = image2[mask]
    mse = np.mean((valid_pixels1 - valid_pixels2) ** 2)
    return mse

def calculate_rmse(image1, image2):
    """Calculate the Root Mean Squared Error between two images."""
    mse = calculate_mse(image1, image2)
    rmse = np.sqrt(mse)
    return rmse

# Example usage
path1 = r"D:\projects\MyCSPN\output\sgd0521_lr1e-3_step6_mynet_layer2_rgb_nosparse_mae_normalize_gradclip_sparsemask_dynamicmask_refinergbedit_noinit_clamp\latest_epoch_result\00002_gt.png"
path2 = r"D:\projects\MyCSPN\output\sgd0521_lr1e-3_step6_mynet_layer2_rgb_nosparse_mae_normalize_gradclip_sparsemask_dynamicmask_refinergbedit_noinit_clamp\latest_epoch_result\00002_pred.png"
image1 = load_image_as_array(path1)
image2 = load_image_as_array(path2)

# Calculate MAE
mae = calculate_mae(image1, image2)
print("The MAE loss between the images is:", mae)

# Calculate MSE
mse = calculate_mse(image1, image2)
print("The MSE loss between the images is:", mse)

# Calculate RMSE
rmse = calculate_rmse(image1, image2)
print("The RMSE loss between the images is:", rmse)
