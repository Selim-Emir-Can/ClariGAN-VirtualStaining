import os
from PIL import Image
import re

# Define the parent folder path and condition folder path
parent_folder = r"C:\Users\ammic\Desktop\ClariGAN-DL\finetuned_dataset_resultsBDE10x10"
condition_folder = os.path.join(parent_folder, 'ground_truth')
part_num = 0
letter = "Z"
# Function to extract row and column numbers from file name
def extract_row_col(file_name):
    # match = re.search(f"part{part_num}_row(\d+)_col(\d+)", file_name)
    match = re.search(f"{letter}_row(\d+)_col(\d+)", file_name)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
        return row, col
    return None, None


# Create a dictionary to hold condition images with row and column as keys
stitched_conditions = {}

# Process all condition images in the condition folder
for file_name in os.listdir(condition_folder):
    if file_name.endswith(".png"):
        row, col = extract_row_col(file_name)
        if row is not None and col is not None:
            file_path = os.path.join(condition_folder, file_name)
            condition_img = Image.open(file_path)
            stitched_conditions[(row, col)] = condition_img

# Determine overall grid size
max_row = max(row for row, col in stitched_conditions.keys())
max_col = max(col for row, col in stitched_conditions.keys())

# Stitch all condition images into the final image
overall_width = max(img.width for img in stitched_conditions.values())
overall_height = max(img.height for img in stitched_conditions.values())
final_condition_image = Image.new('RGB', (overall_width * (max_col + 1), overall_height * (max_row + 1)))

for (row, col), img in stitched_conditions.items():
    final_condition_image.paste(img, (col * overall_width, row * overall_height))

# Save the final stitched condition image
final_condition_image_path = os.path.join(parent_folder, f"part{part_num}_stitched_ground_truth_image.png")
final_condition_image_path = os.path.join(parent_folder, r"stitched_ground_truth_image.png")
final_condition_image.save(final_condition_image_path)
print(f"Stitched condition image saved at: {final_condition_image_path}")
