"""
Writtend by Christopher Dauchter (Erickson)

TODO:
    Allow for the projected image to run in real time
    Add a fullscreen secondary monitor feature, that auto detect the scan bed, maybe use scan lightto help orient
"""


import cv2
import numpy as np
import os
import subprocess
import colorsys
from typing import Tuple

def hsv2rgb(h: float, s: float, v: float) -> Tuple[int, int, int]:
    """
    Convert HSV values to RGB values.

    Args:
        h: Hue value (0.0 - 1.0).
        s: Saturation value (0.0 - 1.0).
        v: Value (brightness) value (0.0 - 1.0).

    Returns:
        Tuple containing the RGB values.

    """
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

def crop_objects(image_path: str, min_object_size: int, max_object_size: int, padding: int) -> None:
    """
    Crop objects from an image based on their size and save them as separate images.

    Args:
        image_path: Path to the grayscale image.
        min_object_size: Minimum object size in pixels.
        max_object_size: Maximum object size in pixels.
        padding: Padding to be added to the cropped images.

    Returns:
        None.

    """
    # Read the grayscale image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to obtain a binary image
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours of white objects in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a directory to save the cropped images
    output_dir = 'cropped_images/'
    os.makedirs(output_dir, exist_ok=True)
    j = 1

    # Crop and save each white object as a separate image
    for i, contour in enumerate(contours):
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Check if object size meets the minimum and maximum requirements
        if min_object_size <= w <= max_object_size and min_object_size <= h <= max_object_size:
            # Add padding to the bounding box coordinates
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding

            # Ensure that the modified bounding box is within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            # Crop the white object from the original grayscale image
            cropped_image = image[y:y + h, x:x + w]

            # Save the cropped image
            cv2.imwrite(f'{output_dir}Gear_{j}.jpg', cropped_image)

            # Call the other Python script with the cropped image path as an argument
            result = subprocess.check_output(['python', 'GearAnal.py', f"{output_dir}Gear_{j}.jpg"])
            print(float(result))
            number = round(float(result), 1)
            #print(number)
            # Apply square colored overlays based on the returned number
            color = hsv2rgb(number, 1, 1)  # Modify the color based on your requirements
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=-1)

            j += 1

    # Save the final image with overlays
    cv2.imwrite('output_image.jpg', image)

# Usage: Provide the path to the grayscale image, minimum object size, maximum object size, and padding
image_path = 'big_image.jpg'
min_object_size = 50
max_object_size = 2000
padding = 60
crop_objects(image_path, min_object_size, max_object_size, padding)
