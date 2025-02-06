import os
from PIL import Image
import re

# Define the parent folder path
save_folder = r"C:\Users\ammic\Desktop\ClariGAN-DL\finetuned_dataset_resultsBDE10x10"
parent_folder = os.path.join(save_folder, "200")
num = "0"
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

# Collect all subfolders and their row/col info
subfolders = []
for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)
    if os.path.isdir(folder_path):
        row, col = extract_row_col(folder_name)
        if row is not None and col is not None:
            subfolders.append((row, col, folder_path))

# Sort subfolders by row and column
subfolders.sort(key=lambda x: (x[0], x[1]))

# Create a dictionary to hold images
stitched_images = {}

# Process each subfolder
for row, col, folder_path in subfolders:
    # Collect all output images in the current folder
    images = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.startswith("output_"+num) and file_name.endswith(".png"):
            file_path = os.path.join(folder_path, file_name)
            images.append(Image.open(file_path))
    
    # Stitch images in the folder horizontally (row-wise)
    if images:
        stitched_row = Image.new('RGB', (sum(img.width for img in images), images[0].height))
        x_offset = 0
        for img in images:
            stitched_row.paste(img, (x_offset, 0))
            x_offset += img.width
        stitched_images[(row, col)] = stitched_row

# Determine overall grid size
max_row = max(row for row, col in stitched_images.keys())
max_col = max(col for row, col in stitched_images.keys())

# Stitch all rows into the final image
overall_width = max(img.width for img in stitched_images.values())
overall_height = max(img.height for img in stitched_images.values())
final_image = Image.new('RGB', (overall_width * (max_col + 1), overall_height * (max_row + 1)))

for (row, col), img in stitched_images.items():
    final_image.paste(img, (col * overall_width, row * overall_height))

# Save the final stitched image
final_image_path = os.path.join(save_folder, f"R3-{letter}_stitched_image"+num+".png")
# final_image_path = os.path.join(save_folder, f"part{part_num}_stitched_image"+num+".png")
final_image.save(final_image_path)
print(f"Stitched image saved at: {final_image_path}")
