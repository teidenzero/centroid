def generate_quadra_subframes(image, centroid, subframe_size=(904, 904)):
    """
    Generate four 904x904 subframes positioned differently around a given centroid.
    Each subframe is adjusted to remain within the original image boundaries.

    Parameters:
    - image: The original image.
    - centroid: The centroid (x, y) around which the subframes are centered.
    - subframe_size: The size of each subframe (width, height).

    Returns:
    - A list of four subframes.
    """
    original_height, original_width = image.shape[:2]
    subframes = []

    # Definitions for each subframe's corner relative to the centroid
    corners = [
        ("lower_right", (centroid[0] - subframe_size[0], centroid[1] - subframe_size[1])),
        ("lower_left", (centroid[0], centroid[1] - subframe_size[1])),
        ("upper_right", (centroid[0] - subframe_size[0], centroid[1])),
        ("upper_left", centroid)
    ]

    for corner_name, (start_x, start_y) in corners:
        # Adjust to ensure the subframe stays within the image boundaries
        end_x = min(start_x + subframe_size[0], original_width)
        end_y = min(start_y + subframe_size[1], original_height)
        adjusted_start_x = max(end_x - subframe_size[0], 0)
        adjusted_start_y = max(end_y - subframe_size[1], 0)

        # Extract the subframe
        subframe = image[adjusted_start_y:end_y, adjusted_start_x:end_x]
        subframes.append((corner_name, subframe))

    return subframes

def save_subframes(subframes, base_filename, output_folder):
    """
    Save each subframe to a specified folder with a name indicating its position relative to the centroid.

    Parameters:
    - subframes: A list of tuples containing the subframe's position name and the subframe image.
    - base_filename: The base name for the output files.
    - output_folder: The folder where subframes should be saved.
    """
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for corner_name, subframe in subframes:
        filename = f"{base_filename}_{corner_name}.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, subframe)

# Usage example:
# original_image = cv2.imread('path_to_image.png')
# centroid = (1920, 1080)  # Example centroid
# quadra_subframes = generate_quadra_subframes(original_image, centroid)
# save_subframes(quadra_subframes, 'subframe', '/path_to_output_folder')
