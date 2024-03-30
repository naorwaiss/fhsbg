import asyncio
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

# Setup logging
log_handler = RotatingFileHandler('position_velocity.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('position_velocity')
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(log_handler)
logger.info("-----------------------------------------")

max_velocity = 2.0  # Initialization of Maximum velocity limit

def calculate_velocity(position, velocity_ratio): # Function to calculate velocity with a custom ratio from position
    return position * velocity_ratio

def adjust_velocity_ratios(z_filtered): # Function to adjust velocity ratios based on z_filtered value
    if z_filtered < 1.0:
        x_velocity_ratio = 0.25
        y_velocity_ratio = 0.25
        z_velocity_ratio = 0.25
    elif 1.0 <= z_filtered < 2.0:
        x_velocity_ratio = 0.5
        y_velocity_ratio = 0.5
        z_velocity_ratio = 0.5
    else:
        x_velocity_ratio = 1.0
        y_velocity_ratio = 1.0
        z_velocity_ratio = 1.0
    return x_velocity_ratio, y_velocity_ratio, z_velocity_ratio

async def process_frames_with_custom_velocity(video_processor_queue, position_velocity_queue):
    try:
        while True:
            # Get position and depth values from the queue
            x_original, y_original, z_original, x_filtered, y_filtered, z_filtered = await video_processor_queue.get()
            # Adjust velocity ratios based on z_filtered value using the separate function
            x_velocity_ratio, y_velocity_ratio, z_velocity_ratio = adjust_velocity_ratios(z_filtered)
            # Calculate velocity with custom ratio for X, Y, and Z axes
            x_velocity_calculate = calculate_velocity(x_filtered, x_velocity_ratio)
            y_velocity_calculate = calculate_velocity(y_filtered, y_velocity_ratio)
            z_velocity_calculate = calculate_velocity(z_filtered, z_velocity_ratio)
            # Apply the maximum velocity limit
            x_velocity = min(max_velocity, x_velocity_calculate)
            y_velocity = min(max_velocity, y_velocity_calculate)
            z_velocity = min(max_velocity, z_velocity_calculate)
            await position_velocity_queue.put((x_velocity, y_velocity, z_velocity))
            message = f"Velocity: {x_velocity:.2f}, {y_velocity:.2f}, {z_velocity:.2f}"
            logger.info(message)
            print(message)
    except asyncio.CancelledError:
        message = "Frame processing has been cancelled."
        logger.warning(message)
        print(message)
    except Exception as e:
        message = f"Unexpected error in frame processing: {e}"
        logger.error(message)
        print(message)

async def print_velocitys(position_velocity_queue):
    try:
        while True:
            x_velocity, y_velocity, z_velocity = await position_velocity_queue.get()
            message = f"Velocity: Vx = {x_velocity:.2f}, Vy = {y_velocity:.2f}, Vz = {z_velocity:.2f}" "-" * 45
            logger.info(message)
            print(message)
    except asyncio.CancelledError:
        message = "Velocity printing has been cancelled."
        logger.warning(message)
        print(message)
    except Exception as e:
        message = f"Unexpected error in velocity printing: {e}"
        logger.error(message)
        print(message)