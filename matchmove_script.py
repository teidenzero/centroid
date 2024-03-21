
import cv2
import numpy as np
import os

# Function to read an image sequence from a directory
def read_image_sequence(directory):
    images = []
    file_names = sorted([f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')])
    for file_name in file_names:
        img = cv2.imread(os.path.join(directory, file_name))
        if img is not None:
            images.append(img)
    return images

# Function to find the centroid of the red shape in an image
def find_red_centroid(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None

# Function to matchmove with reference frame option
def matchmove(sequence1_dir, sequence2_dir, output_dir, centroids_file, reference_frame_number=0):
    sequence1 = read_image_sequence(sequence1_dir)
    sequence2 = read_image_sequence(sequence2_dir)
    centroids = []
    if len(sequence1) != len(sequence2):
        raise ValueError("Image sequences are of different lengths.")
    if reference_frame_number >= len(sequence1) or reference_frame_number < 0:
        raise ValueError("Invalid reference frame number.")
    reference_centroid = find_red_centroid(sequence1[reference_frame_number])
    for i, (img1, img2) in enumerate(zip(sequence1, sequence2)):
        centroid = find_red_centroid(img1)
        if centroid:
            offset = (centroid[0] - reference_centroid[0], centroid[1] - reference_centroid[1])
            M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
            shifted_img = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))
            cv2.imwrite(os.path.join(output_dir, f'modified_{i:04d}.jpg'), shifted_img)
            centroids.append(centroid)
        else:
            centroids.append(centroids[-1] if centroids else (0, 0))
    with open(centroids_file, 'w') as f:
        for cx, cy in centroids:
            f.write(f'{cx},{cy}\n')

# Function to reverse matchmove
def reverse_matchmove(modified_sequence_dir, output_dir, centroids_file):
    modified_sequence = read_image_sequence(modified_sequence_dir)
    centroids = read_centroids(centroids_file)
    if len(modified_sequence) != len(centroids):
        raise ValueError("The image sequence and centroids file are of different lengths.")
    first_centroid = centroids[0]
    for i, (img, centroid) in enumerate(zip(modified_sequence, centroids)):
        offset = (first_centroid[0] - centroid[0], first_centroid[1] - centroid[1])
        M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
        reverted_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        cv2.imwrite(os.path.join(output_dir, f'reverted_{i:04d}.jpg'), reverted_img)

# Example function calls (uncomment to use)
# matchmove('sequence1_dir', 'sequence2_dir', 'output_dir', 'centroids.txt')
# reverse_matchmove('modified_sequence_dir', 'output_dir', 'centroids.txt')
