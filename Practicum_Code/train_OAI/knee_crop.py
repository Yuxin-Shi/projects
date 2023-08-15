import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_knees_center(img_dilation, row_group_size=5, col_group_size=3, row_threshold=0.6, col_threshold=0.3):
    """
    Function to find the center of the knees based on the given dilation image and parameters.
    Returns the row index and the average of top column indices.
    """
    # Function to get the top indices within the threshold
    def get_top_indices_within_threshold(values, threshold, group_size, top_n=6):
        start_index = int(len(values) * ((1 - threshold) / 2))
        end_index = start_index + int(len(values) * threshold)
        middle_values = values[start_index:end_index]
        sorted_indices = np.argsort(middle_values)[::-1]
        top_indices = [(i + start_index) * group_size + group_size // 2 for i in sorted_indices[:top_n]]
        return top_indices

    # Calculate sums of pixel intensities along the rows and columns
    row_sums = [np.sum(img_dilation[i:i+row_group_size]) for i in range(0, img_dilation.shape[0], row_group_size)]
    col_sums = [np.sum(img_dilation[:, i:i+col_group_size]) for i in range(0, img_dilation.shape[1], col_group_size)]

    # Determine row and column indices for the knee center
    row_index = get_top_indices_within_threshold(row_sums, row_threshold, row_group_size, top_n=1)[0]
    col_indices = get_top_indices_within_threshold(col_sums, col_threshold, col_group_size)

    return row_index, int(np.round(np.mean(col_indices)))


def crop_knees(img, center, square_length=220, apart=40):
    """
    Function to crop out the knees from the image based on the center, square length and apart parameters.
    Returns the cropped region, left knee and right knee images.
    """
    row_index, col_index = center
    half_length = square_length // 2
    half_apart = apart // 2

    # Define the boundaries of the rectangular region
    top = row_index - half_length
    bottom = row_index + half_length
    left = col_index - square_length - half_apart
    right = col_index + square_length + half_apart

    # Crop the rectangular region from the image
    region = img[top:bottom, left:right]

    # Define the boundaries of the left and right squares within the region
    left_knee = region[:, :square_length]
    right_knee = region[:, square_length + apart:]

    return region, left_knee, right_knee


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', default="C:/University/2022_Fall/CHL5207/Knee_Osteoarthritis/OAI_raw/100_OAI")
    parser.add_argument('--crop_img_dir', default="C:/University/2022_Fall/CHL5207/Knee_Osteoarthritis/OAI_raw/knee_crop")
    args = parser.parse_args()
    # Loop over all .jpg files in the current directory
    for img_path in os.listdir(args.image_dir):
        # Continue to the next file if current file is not a .jpg file
        if not img_path.endswith('.jpg'):
            continue

        # Load the image
        img = cv2.imread(os.path.join(args.image_dir, img_path))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform histogram equalization
        equ = cv2.equalizeHist(gray)

        # Perform edge detection
        edges = cv2.Canny(equ, 100, 200)

        # Perform erosion and dilation
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(edges, kernel, iterations=1)

        # Determine the center of the knees
        center = find_knees_center(img_dilation)

        # Crop the knees from the image
        _, left_knee, right_knee = crop_knees(img, center)

        # Save the cropped knee images to the pre-specified directory
        try:
            cv2.imwrite(os.path.join(args.crop_img_dir, os.path.splitext(img_path)[0] + '_R.png'), left_knee)
            cv2.imwrite(os.path.join(args.crop_img_dir, os.path.splitext(img_path)[0] + '_L.png'), right_knee)
        except:
            print(os.path.splitext(img_path)[0])
            pass
