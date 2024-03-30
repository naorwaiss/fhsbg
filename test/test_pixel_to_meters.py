import numpy as np

# Define the pixel_to_meters function
def pixel_to_meters(x_pixel, y_pixel, fov_horizontal, fov_vertical, image_width, image_height, distance_to_object):
    angle_per_pixel_x = fov_horizontal / image_width
    angle_per_pixel_y = fov_vertical / image_height
    angle_offset_x = (x_pixel - (image_width / 2)) * angle_per_pixel_x
    angle_offset_y = ((image_height / 2) - y_pixel) * angle_per_pixel_y
    angle_offset_x_radians = np.radians(angle_offset_x)
    angle_offset_y_radians = np.radians(angle_offset_y)
    tan_x = np.tan(angle_offset_x_radians)
    tan_y = np.tan(angle_offset_y_radians)
    x_meters = tan_x * distance_to_object
    y_meters = tan_y * distance_to_object
    return x_meters, y_meters


# Assuming you have calibration data as follows:
# Each item: (x_pixel, y_pixel, real_world_x_meters, real_world_y_meters, distance_to_object)
calibration_data = [
    (960, 540, 0, 0, 2.0),  # Example data point
    # Add more data points here
]

# Function to test calibration data against the pixel_to_meters function
def test_calibration(data, fov_horizontal, fov_vertical, image_width, image_height):
    errors = []
    for x_pixel, y_pixel, expected_x, expected_y, distance in data:
        calculated_x, calculated_y = pixel_to_meters(x_pixel, y_pixel, fov_horizontal, fov_vertical, image_width, image_height, distance)
        error_x = calculated_x - expected_x
        error_y = calculated_y - expected_y
        errors.append((error_x, error_y))
    return errors

# Initial FOV values
fov_horizontal = 69
fov_vertical = 42

# Test the calibration and adjust FOVs as needed
errors = test_calibration(calibration_data, fov_horizontal, fov_vertical, 1920, 1080)
print("Calibration Errors:", errors)

# Based on errors, adjust fov_horizontal and fov_vertical
# This requires manual analysis and adjustments





# Example usage
if __name__ == "__main__":
    # Assuming some values for testing
    fov_horizontal = 69  # Horizontal field of view in degrees
    fov_vertical = 42  # Vertical field of view in degrees
    image_width = 1920  # Image width in pixels
    image_height = 1080  # Image height in pixels
    distance_to_object = 2.0  # Distance to the object in meters

    # Pixel coordinates of the object
    x_pixel = 0  # Pixel X coordinate (in the center of the image for this example)
    y_pixel = 0  # Pixel Y coordinate (in the center of the image for this example)

    # Call the function
    x_meters, y_meters = pixel_to_meters(x_pixel, y_pixel, fov_horizontal, fov_vertical, image_width, image_height,
                                         distance_to_object)

    # Print the results
    print(f"X meters: {x_meters}, Y meters: {y_meters}")
