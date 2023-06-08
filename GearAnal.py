"""
Writtend by Christopher Dauchter (Erickson)

TODO:
    Make parameters commandline arguements
    Make a better outer circle detection
    Improve runtime
    Dynamic Parameters on failure?
"""


import cv2
import numpy as np
import argparse
import math
from typing import Tuple, List

def count_teeth(center: Tuple[float, float], outer_radius: float, inner_radius: float,
                binary_image: np.ndarray, resolution: int) -> Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Count the number of teeth in a gear.

    Args:
        center: The center coordinates of the gear.
        outer_radius: The outer radius of the gear.
        inner_radius: The inner radius of the gear.
        binary_image: The binary image of the gear.
        resolution: The resolution or number of angular divisions to evaluate.

    Returns:
        num_teeth: The number of teeth in the gear.
        tooth_positions: The positions of the teeth.
        search_points: The points used for searching teeth.

    """
    mid_radius = (outer_radius + inner_radius) / 2

    # Initial point
    initial_point = (center[0] + mid_radius, center[1])

    # Number of angular divisions to evaluate
    num_divisions = resolution

    num_teeth = 0
    last_pixel_value = 0
    tooth_positions = []
    search_points = []
    for i in range(1, num_divisions + 1):
        # Calculate the angle
        angle = i * ((2 * math.pi) / num_divisions)

        # Calculate the new point
        px = center[0] + mid_radius * math.cos(angle)
        py = center[1] + mid_radius * math.sin(angle)
        p = (px, py)

        # Add the point to the search points list
        search_points.append((int(p[0]), int(p[1])))

        # Check if the point is within the image boundaries
        pixel_value = binary_image[int(p[1]), int(p[0])]

        # If the pixel value has changed, increment the tooth count
        if pixel_value != last_pixel_value:
            num_teeth += 1
            last_pixel_value = pixel_value
            tooth_positions.append((int(p[0]), int(p[1])))

    # Divide the total by 2 to count every 2 changes as a tooth
    num_teeth = math.floor(num_teeth / 2)

    return num_teeth, tooth_positions, search_points


def find_secondary_circle(center: Tuple[float, float], outer_radius: float, binary_image: np.ndarray,
                          max_iterations: int, given_resolution: int) -> float:
    """
    Find the secondary circle in a gear.

    Args:
        center: The center coordinates of the gear.
        outer_radius: The outer radius of the gear.
        binary_image: The binary image of the gear.
        max_iterations: The maximum number of iterations for the secondary circle search.
        given_resolution: The resolution or number of angular divisions.

    Returns:
        radius: The radius of the secondary circle.

    """
    radius = outer_radius
    num_teeth = 0
    delta = 1
    for i in range(max_iterations):
        radius -= delta
        if radius <= 1:
            radius = 1
        _, teeth_points, _ = count_teeth(center, radius + delta, radius, binary_image, given_resolution)
        num_teeth_new = len(teeth_points)
        if num_teeth_new <= num_teeth:
            break

    return radius

def process_gear_image(image_path: str, scandpi: int, kernel_lim: int, blur_kernel_size: Tuple[int, int],
                       threshold_param1: int, threshold_param2: int, min_radius_factor: float,
                       max_radius_factor: float, max_secondary_search_iterations: int, resolution: int) -> float:
    """
    Process a gear image and calculate the module number.

    Args:
        image_path: Path to the gear image.
        scandpi: The scanning dpi.
        kernel_lim: The kernel limit for Gaussian blur.
        blur_kernel_size: The kernel size for Gaussian blur.
        threshold_param1: Param1 for adaptive thresholding.
        threshold_param2: Param2 for adaptive thresholding.
        min_radius_factor: Minimum radius factor.
        max_radius_factor: Maximum radius factor.
        max_secondary_search_iterations: Maximum number of iterations for secondary circle search.
        resolution: The resolution or number of angular divisions.

    Returns:
        The calculated module number.

    """
    # Load the gear image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a blank image for displaying the processed image
    output_image = image.copy()

    # Apply Gaussian blur to reduce noise for finding the outer radius
    blurred_circle = cv2.GaussianBlur(gray, blur_kernel_size, 0)

    # Apply adaptive thresholding to create a binary image for finding the outer radius
    _, binary_circle = cv2.threshold(blurred_circle, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define the expected radius range based on the image size for finding the outer radius
    height_circle, width_circle = binary_circle.shape[:2]
    min_radius_circle = int(min(height_circle, width_circle) * min_radius_factor)
    max_radius_circle = int(min(height_circle, width_circle) * max_radius_factor)

    # Apply Hough Circle Transform on a limited ROI to find the outer radius
    circles = cv2.HoughCircles(
        binary_circle, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(min(height_circle, width_circle) * min_radius_factor),
        param1=threshold_param1, param2=threshold_param2,
        minRadius=min_radius_circle, maxRadius=max_radius_circle
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        outer_circle = circles[0][0]  # First detected circle is the outer circle
        center_circle = (outer_circle[0], outer_circle[1])  # Center coordinates in resized image
        radius_circle = outer_circle[2]  # Radius in resized image

        tooth_blur_kernel_size = (math.floor(kernel_lim / 2), math.floor(kernel_lim / 2))
        tooth_blurred_circle = cv2.GaussianBlur(gray, tooth_blur_kernel_size, 0)

        # Find the secondary circle
        secondary_radius = find_secondary_circle(center_circle, radius_circle, binary_circle,
                                                 max_secondary_search_iterations, resolution)

        # Apply adaptive thresholding to create a binary image for finding the teeth
        _, binary_teeth_roi = cv2.threshold(tooth_blurred_circle, 0, 255, cv2.THRESH_OTSU)

        num_teeth, tooth_positions, search_points = count_teeth(center_circle, radius_circle, secondary_radius,
                                                                binary_teeth_roi, resolution)

        # Find the midpoint circle on the output image
        midpoint_radius = (radius_circle + secondary_radius) // 2
        CalcModule = round(((2 * midpoint_radius) / (num_teeth * (scandpi / 25.4))), 3)

        return CalcModule
    else:
        return print("No circle found. Adjust Hough Circle Transform parameters or check the image quality.")

def main() -> None:
    """
    Main function to run the gear image analysis.
    """
    # Adjustable parameters
    scandpi = 1200
    kernel_lim = 55
    blur_kernel_size = (kernel_lim, kernel_lim)  # Kernel size for Gaussian blur
    threshold_param1 = 1000  # Param1 for adaptive thresholding
    threshold_param2 = 1  # Param2 for adaptive thresholding
    min_radius_factor = 0.13  # Minimum radius factor
    max_radius_factor = 1  # Maximum radius factor
    max_secondary_search_iterations = 60  # Maximum number of iterations for secondary circle search
    resolution = 3000

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process a gear image and calculate the module number.')
    parser.add_argument('image_path', type=str, help='path to the gear image')
    args = parser.parse_args()

    # Run the gear image processing
    module_number = process_gear_image(args.image_path, scandpi, kernel_lim, blur_kernel_size, threshold_param1,
                                       threshold_param2, min_radius_factor, max_radius_factor,
                                       max_secondary_search_iterations, resolution)

    # Print the module number
    print(module_number)

if __name__ == '__main__':
    main()
