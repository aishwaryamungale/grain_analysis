from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import cv2
import os
import json

# Load the FastSAM model
model = FastSAM('./weights/FastSAM-x.pt')

# Specify the root directory
root_directory = r'\data'

# Set the device (CPU in this case)
DEVICE = 'cpu'


# Iterate through each subfolder in the input path
for root, dirs, files in os.walk(root_directory):
    for file in files:
        # Check if the file is an image (you can customize the list of supported image extensions)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct the input file path
            image_path = os.path.join(root, file)

            #increase_contrast(image_path, image_path)

            # Process the image with FastSAM
            everything_results = model(image_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
            prompt_process = FastSAMPrompt(image_path, everything_results, device=DEVICE)

            # Get bounding box annotations
            ann = prompt_process.everything_prompt()

            # Load the original image
            original_image = cv2.imread(image_path)

            # List to store the bounding box coordinates
            bounding_boxes = []

            for i, mask in enumerate(ann):
                mask_numpy = (mask.cpu().numpy() * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for j, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    # Append the bounding box coordinates to the list
                    bounding_boxes.append({
                        'mask_index': i + 1,
                        'contour_index': j + 1,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    })
                    # Crop the region from the original image
                    cropped_region = original_image[y:y + h, x:x + w]

                    # Save the cropped image in the same subfolder
                    output_filename = f'{os.path.splitext(file)[0]}_cropped_{i + 1}_contour_{j + 1}.jpg'
                    output_path = os.path.join(root, output_filename)
                    cv2.imwrite(output_path, cropped_region)

            # Save bounding box information to a JSON file in the same subfolder
            output_file = os.path.join(root, 'bounding_boxes.json')
            with open(output_file, 'w') as json_file:
                json.dump(bounding_boxes, json_file)

            print(f'Bounding box information saved at: {output_file}')
