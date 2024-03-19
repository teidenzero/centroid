import cv2
import numpy as np
import os
import argparse

# Implementing the requested functionalities

def write_centroids_to_file(centroids, file_path):
    """
    Writes centroid coordinates to a text file.

    Parameters:
    - centroids: List of tuples representing the centroid coordinates.
    - file_path: Path to the output text file.
    """
    with open(file_path, 'w') as f:
        for centroid in centroids:
            f.write(f'{centroid[0]},{centroid[1]}\n')

def read_centroids_from_file(file_path):
    """
    Reads centroid coordinates from a text file.

    Parameters:
    - file_path: Path to the text file containing centroid coordinates.

    Returns:
    - List of tuples representing the centroid coordinates.
    """
    centroids = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y = line.strip().split(',')
            centroids.append((int(x), int(y)))
    return centroids

def composite_subframes_on_directory(original_images_dir, subframes_dir, centroids_file, output_dir, portrait):
    """
    For each original image in the specified directory, composite the corresponding subframe
    based on centroid coordinates from the text file and save the composited image.

    Parameters:
    - original_images_dir: Directory containing the original images.
    - subframes_dir: Directory containing the subframe images.
    - centroids_file: File containing the centroid coordinates for each image.
    - output_dir: Directory where the composited images will be saved.
    """
    # Read the centroids from the file
    centroids = read_centroids_from_file(centroids_file)
    
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subframes = []
    i = 0
    for subname in os.listdir(subframes_dir):
        print(subname)
        subframes.append(subname)
    print(subframes[0])



    # Process each original image
    for idx, filename in enumerate(sorted(os.listdir(original_images_dir))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_image_path = os.path.join(original_images_dir, filename)
            original_image = cv2.imread(original_image_path)
            original_height, original_width = original_image.shape[:2]

            # Assume subframes and centroids correspond in order
            subframe_path = os.path.join(subframes_dir, subframes[idx])  # Update naming convention if needed
            centroid = centroids[idx]

            

            # Calculate where to place the subframe
            subframe = cv2.imread(subframe_path)
            if portrait == True:
                subframe = cv2.rotate(subframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
            subframe_height, subframe_width = subframe.shape[:2]
            # Calculate top-left corner of the subframe ensuring it doesn't go out of the original image boundaries
            top_left_x = max(min(centroid[0] - subframe_width // 2, original_width - subframe_width), 0)
            top_left_y = max(min(centroid[1] - subframe_height // 2, original_height - subframe_height), 0)

            # Place the subframe onto the original image
            original_image[top_left_y:top_left_y + subframe_height, top_left_x:top_left_x + subframe_width] = subframe

            # Save the composited image
            output_path = os.path.join(output_dir, f"composited_{filename}")
            cv2.imwrite(output_path, original_image)
        



def find_red_shape_centroid(image_path):
    """
    Process the given image to find the centroid of the largest red shape.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Tuple (x, y) of centroid coordinates, or None if no red shape is found.
    """
    print('red shape started')
    image = cv2.imread(image_path)  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    print(mask1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)
    if contours:
        print('contours')
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            return centroid_x, centroid_y
    return None

def process_folder(folder_path):
    """
    Process all images in the given folder to find and print the centroid of the largest red shape.

    Parameters:
    - folder_path: Path to the folder containing image files.
    """
    centroids = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, file_name)
            centroid = find_red_shape_centroid(image_path)
            centroids[file_name] = centroid
    return centroids

def extract_subframe(image, centroid, subframe_size=(1920, 1080), original_size=(3840, 2160)):
    """
    Extract a subframe centered on the centroid from the original image,
    adjusting for bounds if necessary.

    Parameters:
    - image: The original image.
    - centroid: Tuple (x, y) of the centroid coordinates.
    - subframe_size: Size of the subframe as (width, height).
    - original_size: Size of the original image as (width, height).

    Returns:
    - The extracted subframe.
    """
    x_center, y_center = centroid
    sub_width, sub_height = subframe_size
    original_width, original_height = original_size

    # Calculate half sizes for easier boundary checks
    half_sub_width = sub_width // 2
    half_sub_height = sub_height // 2

    # Initialize subframe coordinates
    left = x_center - half_sub_width
    right = x_center + half_sub_width
    top = y_center - half_sub_height
    bottom = y_center + half_sub_height

    # Adjust for bounds on the x-axis
    if left < 0:
        left = 0
        right = sub_width
    elif right > original_width:
        right = original_width
        left = original_width - sub_width

    # Adjust for bounds on the y-axis
    if top < 0:
        top = 0
        bottom = sub_height
    elif bottom > original_height:
        bottom = original_height
        top = original_height - sub_height

    # Extract and return the subframe
    subframe = image[top:bottom, left:right]
    return subframe

def save_subframe(subframe, file_name):
    """
    Save the given subframe to a file.

    Parameters:
    - subframe: The subframe to save.
    - file_name: Name of the file to save the subframe.
    """
    cv2.imwrite(file_name, subframe)

def process_images_and_extract_subframes(plate_folder_path, plate_output_folder, mask_folder_path, mask_output_folder, erode, portrait):
    """
    Process all images in the given folder, extract subframes centered on the centroids of red shapes,
    and save the subframes to the specified output folder.

    Parameters:
    - folder_path: Path to the folder containing image files.
    - output_folder: Path to the folder where subframes should be saved.
    """
    if not os.path.exists(plate_output_folder):
        os.makedirs(plate_output_folder)
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    centroids = []
    i = 0

    for file_name_mask in os.listdir(mask_folder_path):
        if file_name_mask.lower().endswith(('.png', '.jpg', '.jpeg')):
            mask_path = os.path.join(mask_folder_path, file_name_mask)
            print(mask_path)            

            
            centroid = find_red_shape_centroid(mask_path)
            
            centroids.append(centroid)
            if centroid:
                print('centroid OK mask')
                mask = cv2.imread(mask_path)
                subframe = extract_subframe(mask, centroid)
                if erode == True:
                    subframe = erode_subframe_and_get_thick_contour_image(subframe, 10, 20)
                output_path = os.path.join(mask_output_folder, f"subframe_{file_name_mask}")
                if portrait == True:
                    subframe = cv2.rotate(subframe, cv2.ROTATE_90_CLOCKWISE)
                save_subframe(subframe, output_path)

    for file_name_plate in os.listdir(plate_folder_path):
        if file_name_plate.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(plate_folder_path, file_name_plate)
            centroid = centroids[i]
            i = i + 1
            if centroid:
                print('centroid OK plate')
                image = cv2.imread(img_path)
                subframe = extract_subframe(image, centroid)
                output_path = os.path.join(plate_output_folder, f"subframe_{file_name_plate}")
                if portrait == True:
                    subframe = cv2.rotate(subframe, cv2.ROTATE_90_CLOCKWISE)
                save_subframe(subframe, output_path)
    write_centroids_to_file(centroids, plate_output_folder+'\centroids.txt')

# Integration into main function and argparse for command-line execution would follow similar patterns
# as discussed earlier, including the addition of an argument for the output folder.
                
def erode_subframe_and_get_thick_contour_image(subframe, erosion_pixels=5, contour_thickness=3):
    """
    Erode the given subframe by a specified number of pixels and then find and draw thick contours
    on an image, representing the eroded edges.

    Parameters:
    - subframe: The subframe to be eroded.
    - erosion_pixels: The number of pixels by which to erode the subframe.
    - contour_thickness: The thickness of the contour lines in the resulting image.

    Returns:
    - An image with the thick contours highlighted on a black background.
    """
    kernel = np.ones((erosion_pixels*2+1, erosion_pixels*2+1), dtype=np.uint8)
    eroded_subframe = cv2.erode(subframe, kernel, iterations=1)

    # Convert eroded subframe to grayscale and threshold to binary image for contour detection
    eroded_gray = cv2.cvtColor(eroded_subframe, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(eroded_gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black image with the same dimensions as the subframe
    contour_image = np.zeros_like(subframe)

    # Draw the contours with specified thickness
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), contour_thickness)

    return contour_image

# This function finds contours on the eroded subframe and draws them with a specified thickness,
# allowing for a more visually substantial representation of the erosion's impact.


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images to find centroids of red shapes.")
    parser.add_argument("--plate_folder_path", type=str, help="Path to the folder containing image files")
    parser.add_argument("--plate_output_path", type=str, help="Path to the output folder")
    parser.add_argument("--mask_folder_path", type=str, help="Path to the folder containing mask files")
    parser.add_argument("--mask_output_path", type=str, help="Path to the output folder")
    parser.add_argument('--composite', type=str)
    parser.add_argument('--centroids', type=str)
    parser.add_argument('--subframes', type=str)
    parser.add_argument('--composite_output', type=bool)
    parser.add_argument('--portrait', type=bool)
    parser.add_argument('--erode_mask', type=bool)

    # Parse arguments
    args = parser.parse_args()

    erode = False
    portrait = False

    # Process the folder and print centroids
    #centroids = process_folder(args.folder_path)
    if args.erode_mask == True:
            erode = True
    if args.portrait == True:
            portrait = True

    if args.composite == True:
        #centroids = read_centroids_from_file(args.centroids)
        
        composite_subframes_on_directory(args.plate_folder_path, args.subframes, args.centroids, args.composite_output, portrait)
    else:
        
        process_images_and_extract_subframes(args.plate_folder_path, args.plate_output_path, args.mask_folder_path, args.mask_output_path, erode, portrait)
    
    
    #for file_name, centroid in centroids.items():
    #    print(f"{file_name}: {centroid}")

if __name__ == "__main__":
    main()

# Example usage (commented out to avoid execution without a specific folder path)
# folder_path = 'path_to_your_folder'
# centroids = process_folder(folder_path)
# print(centroids)


