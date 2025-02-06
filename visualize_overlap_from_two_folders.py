# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def find_and_plot_images(folder1, folder2, output_folder):
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Get list of files in both folders
#     files1 = sorted(os.listdir(folder1))
#     files2 = sorted(os.listdir(folder2))

#     # Filter for R1 and R3 formatted files
#     r1_files = sorted(f for f in files1 if f.startswith('R1-') and f.endswith('.png'))
#     r3_files = sorted(f for f in files2 if f.startswith('R3-') and f.endswith('.png'))#and f.endswith('_registered_manual_TPS.png'))

#     # Find matching letters
#     matching_letters = {f[3:4] for f in r1_files} & {f[3:4] for f in r3_files}


#     for letter in sorted(matching_letters):
#         # Read the images
#         r1_path = os.path.join(folder1, f'R1-{letter}.png')
#         r3_path = os.path.join(folder2, f'R3-{letter}_registered_manual_TPS.png')

#         r1_image = cv2.imread(r1_path, cv2.IMREAD_COLOR)
#         r3_image = cv2.imread(r3_path, cv2.IMREAD_COLOR)

#         if r1_image is None or r3_image is None:
#             print(f"Error reading images: {r1_path} or {r3_path}")
#             continue

#         # Convert images from BGR to RGB for plotting
#         r1_image = cv2.cvtColor(r1_image, cv2.COLOR_BGR2RGB)
#         r3_image = cv2.cvtColor(r3_image, cv2.COLOR_BGR2RGB)

#         # Ensure images are the same size
#         if r1_image.shape != r3_image.shape:
#             height = min(r1_image.shape[0], r3_image.shape[0])
#             width = min(r1_image.shape[1], r3_image.shape[1])
#             r1_image = cv2.resize(r1_image, (width, height), interpolation=cv2.INTER_AREA)
#             r3_image = cv2.resize(r3_image, (width, height), interpolation=cv2.INTER_AREA)

#         # Compute the overlap (e.g., average of both images)
#         overlap = cv2.addWeighted(r1_image, 0.5, r3_image, 0.5, 0)

#         # Plot the images
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.imshow(r1_image)
#         plt.title(f'R1-{letter}')
#         plt.axis('off')

#         plt.subplot(1, 3, 2)
#         plt.imshow(r3_image)
#         plt.title(f'R3-{letter}')
#         plt.axis('off')

#         plt.subplot(1, 3, 3)
#         plt.imshow(overlap)
#         plt.title(f'Overlap-{letter}')
#         plt.axis('off')

#         plt.tight_layout()

#         # Save the plot
#         output_path = os.path.join(output_folder, f'plot-{letter}.png')
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.close()

#         print(f"Saved plot: {output_path}")

# # Example usage
# folder1 = r"C:\Users\ammic\Downloads\R1-zproj"
# folder2 = r"C:\Users\ammic\Downloads\R3-zproj"
# output_folder = r"C:\Users\ammic\Downloads\R1-R3-overlap"
# find_and_plot_images(folder1, folder2, output_folder)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_and_plot_images(folder1, folder2, output_folder, prefix1="R1-", prefix2="R3-", suffix1=".png", suffix2="_registered_manual_TPS.png"):
    """
    Find and plot images from two folders based on matching patterns and save the plots.

    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.
        output_folder (str): Path to the folder where plots will be saved.
        prefix1 (str): Prefix for images in folder1 (e.g., "R1-").
        prefix2 (str): Prefix for images in folder2 (e.g., "R3-").
        suffix1 (str): Suffix for images in folder1 (e.g., ".png").
        suffix2 (str): Suffix for images in folder2 (e.g., "_registered_manual_TPS.png").
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of files in both folders
    files1 = sorted(os.listdir(folder1))
    files2 = sorted(os.listdir(folder2))

    # Filter files based on prefix and suffix
    r1_files = sorted(f for f in files1 if f.startswith(prefix1) and f.endswith(suffix1))
    r3_files = sorted(f for f in files2 if f.startswith(prefix2) and f.endswith(suffix2))

    # Extract the matching identifier (e.g., a letter or number after the prefix)
    matching_identifiers = {f[len(prefix1):len(prefix1)+1] for f in r1_files} & {f[len(prefix2):len(prefix2)+1] for f in r3_files}

    for identifier in sorted(matching_identifiers):
        # Construct the file paths
        r1_path = os.path.join(folder1, f'{prefix1}{identifier}{suffix1}')
        r3_path = os.path.join(folder2, f'{prefix2}{identifier}{suffix2}')

        # Read the images
        r1_image = cv2.imread(r1_path, cv2.IMREAD_COLOR)
        r3_image = cv2.imread(r3_path, cv2.IMREAD_COLOR)

        if r1_image is None or r3_image is None:
            print(f"Error reading images: {r1_path} or {r3_path}")
            continue

        # Convert images from BGR to RGB for plotting
        r1_image = cv2.cvtColor(r1_image, cv2.COLOR_BGR2RGB)
        r3_image = cv2.cvtColor(r3_image, cv2.COLOR_BGR2RGB)

        # Ensure images are the same size
        if r1_image.shape != r3_image.shape:
            height = min(r1_image.shape[0], r3_image.shape[0])
            width = min(r1_image.shape[1], r3_image.shape[1])
            r1_image = cv2.resize(r1_image, (width, height), interpolation=cv2.INTER_AREA)
            r3_image = cv2.resize(r3_image, (width, height), interpolation=cv2.INTER_AREA)

        # Compute the overlap (e.g., average of both images)
        overlap = cv2.addWeighted(r1_image, 0.5, r3_image, 0.5, 0)

        # Plot the images
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(r1_image)
        plt.title(f'{prefix1}{identifier}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(r3_image)
        plt.title(f'{prefix2}{identifier}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlap)
        plt.title(f'Overlap-{identifier}')
        plt.axis('off')

        plt.tight_layout()

        # Save the plot
        output_path = os.path.join(output_folder, f'plot-{identifier}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved plot: {output_path}")

# Example usage
folder1 = r"C:\Users\ammic\Downloads\Emir-zproj-for-overlap" #r"C:\Users\ammic\Downloads\R1-zproj"
folder2 = folder1 # r"C:\Users\ammic\Downloads\R3-zproj"
output_folder = r"C:\Users\ammic\Downloads\R1-R3-overlap"
find_and_plot_images(folder1, folder2, output_folder, prefix1="R1-", prefix2="R3-", suffix1="_z-proj.png", suffix2="_z-proj.png")
