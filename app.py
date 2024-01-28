from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import cv2
import os

# Load the FastSAM model
model = FastSAM('./weights/FastSAM-x.pt')

# Specify the input image path
input_image_path = r'C:\Users\Aishwarya\Downloads\fastSAM\1.jpeg'

# Specify the output folder
output_folder = r'C:\Users\Aishwarya\Downloads\fastSAM\output'

# Set the device (CPU in this case)
DEVICE = 'cpu'

# Specify the size of the cropped segments
segment_size = 256  # You can adjust this size as needed

# Create output/crops folder if it doesn't exist
crops_folder = os.path.join(output_folder, 'crops')
os.makedirs(crops_folder, exist_ok=True)

# Process the image with FastSAM
everything_results = model(input_image_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
prompt_process = FastSAMPrompt(input_image_path, everything_results, device=DEVICE)

# Get bounding box annotations
ann = prompt_process.everything_prompt()

# Load the original image
original_image = cv2.imread(input_image_path)

# Iterate through the masks and save the crops
for i, mask in enumerate(ann):
    mask_numpy = (mask.cpu().numpy() * 255).astype(np.uint8)
    coords_y, coords_x = np.nonzero(mask_numpy)

    min_y, min_x = np.min(coords_y), np.min(coords_x)
    max_y, max_x = np.max(coords_y), np.max(coords_x)

    h = max_y - min_y
    w = max_x - min_x

    # Get x, y, h, w by doing min-max coords_x and coords_y
    crop = original_image[min_y:min_y+h, min_x:min_x+w].copy()
    crop_mask = mask_numpy[min_y:min_y+h, min_x:min_x+w].astype(bool)
    temp = crop
    temp[~crop_mask] = 0

    # Resize the image of the cropped segment to 128x128
    resized_segment = cv2.resize(temp, (128, 128))

    # Save the resized image in the output/crops folder
    output_filename = f'{os.path.splitext(os.path.basename(input_image_path))[0]}_cropped_{i + 1}.jpg'
    output_path = os.path.join(crops_folder, output_filename)
    cv2.imwrite(output_path, resized_segment)