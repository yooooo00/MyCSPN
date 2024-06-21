import cv2
import os

def images_to_video(image_folder, output_video, fps=10):
    # Get all image files from the folder
    images = [img for img in sorted(os.listdir(image_folder)) if img.split('.')[0].split('_')[-1]=='pred']
    images.sort()  # Sort images by name
    
    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video.write(frame)
    
    # Release the VideoWriter object
    video.release()
    print(f'Video saved as {output_video}')

# Example usage:
images_to_video(r'D:\projects\MyCSPN\output\sgd0526_step6_kitti_uint8_refinergb_layerfull\latest_epoch_result', 'output_video2.mp4')
