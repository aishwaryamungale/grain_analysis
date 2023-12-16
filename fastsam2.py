from fastsam import FastSAM, FastSAMPrompt
import numpy as np
import cv2
import os
import json

# Load the FastSAM model
model = FastSAM('./weights/FastSAM-x.pt')

# Specify the input image path
input_image_path = r'img.jpeg'

# Specify the output folder
output_folder = r'\output'

# Set the device (CPU in this case)
DEVICE = 'cpu'

# Specify the size of the cropped segments
segment_size = 256  # You can adjust this size as needed

# Load the original image
original_image = cv2.imread(input_image_path)
height, width, _ = original_image.shape

# Process the image in segments
for y in range(0, height, segment_size):
    for x in range(0, width, segment_size):
        # Calculate the coordinates for the current segment
        x_end = min(x + segment_size, width)
        y_end = min(y + segment_size, height)

        # Crop the segment from the original image
        segment = original_image[y:y_end, x:x_end]

        # Save the segment as a temporary image file
        temp_segment_path = os.path.join(output_folder, f'temp_segment_{y}_{x}.jpg')
        cv2.imwrite(temp_segment_path, segment)

        # Process the segment with FastSAM
        everything_results = model(temp_segment_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(temp_segment_path, everything_results, device=DEVICE)

        # Get bounding box annotations for the segment
        ann = prompt_process.everything_prompt()

        # List to store the bounding box coordinates for the segment
        bounding_boxes = []

        for i, mask in enumerate(ann):
            mask_numpy = (mask.cpu().numpy() * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_numpy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                x_rel, y_rel, w, h = cv2.boundingRect(contour)
                # Convert relative coordinates to absolute coordinates
                x_abs = x + x_rel
                y_abs = y + y_rel
                # Append the bounding box coordinates to the list
                bounding_boxes.append({
                    'mask_index': i + 1,
                    'contour_index': j + 1,
                    'x': x_abs,
                    'y': y_abs,
                    'width': w,
                    'height': h
                })
                # Crop the region from the original image
                cropped_region = original_image[y_abs:y_abs + h, x_abs:x_abs + w]

                # Save the cropped image in the output folder
                output_filename = f'{os.path.splitext(os.path.basename(input_image_path))[0]}_cropped_{i + 1}_contour_{j + 1}.jpg'
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, cropped_region)

        # Remove the temporary segment image file
        os.remove(temp_segment_path)

# Save bounding box information to a JSON file in the output folder
output_file = os.path.join(output_folder, 'bounding_boxes.json')
with open(output_file, 'w') as json_file:
    json.dump(bounding_boxes, json_file)

print(f'Bounding box information saved at: {output_file}')
