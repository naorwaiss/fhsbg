import asyncio
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import os
import time
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
log_handler = RotatingFileHandler('video_processor.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('video_processor')
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(log_handler)
logger.info("-----------------------------------------")

class Config:
    def __init__(self):
        self.camera_type = '435i'
        self.set_values()
        self.config_dict = {
            'color_range': {
                'lower_red1': np.array([0, 100, 100]),
                'upper_red1': np.array([10, 255, 255]),
                'lower_red2': np.array([160, 100, 100]),
                'upper_red2': np.array([180, 255, 255])
            },
            'video_settings': {
                'filename': os.path.join(os.path.expanduser('~'), 'Desktop', 'output.avi'),
                'fourcc': cv2.VideoWriter_fourcc(*'MJPG'),
                'fps': 30.0,
                'frameSize': (640, 480)  # Set for display resolution
            }
        }
        self.pipeline_config = rs.config()
        self.set_pipeline_config()

    def set_pipeline_config(self):
        # Enable streams at native resolutions
        self.pipeline_config.enable_stream(rs.stream.depth, 0, 1280, 720, rs.format.z16, 30)
        self.pipeline_config.enable_stream(rs.stream.color, 0, 1280, 720, rs.format.bgr8, 30)

    def set_values(self):
        # Field of view adjustments can be made here based on camera specs
        if self.camera_type == '435i':
            self.fov_horizontal = 69
            self.fov_vertical = 42
        elif self.camera_type == '515':
            self.fov_horizontal = 70
            self.fov_vertical = 43
        else:
            logging.error("Invalid camera type specified. Only '435' and '515' are supported.")
            raise ValueError("Invalid camera type. Supported types: '435', '515'.")

class LowPassFilter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.filtered_value = None

    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value  # Initialize with the first value if not already done
        else:
            # Apply the EMA formula
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value

class TrackedObject:
    def __init__(self, contour):
        self.contour = contour
        self.timestamp = time.time()  # Record the time when the object was first detected

async def pixel_to_meters(x_pixel, y_pixel, fov_horizontal, fov_vertical, image_width, image_height, distance_to_object):
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

async def find_largest_obstacle(contours):
    return max(contours, key=cv2.contourArea, default=None)

async def process_frames(video_processor_queue, config, filter_config):
    pipeline = rs.pipeline()
    config_obj = config()

    pipeline_started = False
    tracked_objects = []  # List to keep track of objects

    x_filter = LowPassFilter(filter_config.alpha)
    y_filter = LowPassFilter(filter_config.alpha)
    z_filter = LowPassFilter(filter_config.alpha)

    try:
        # Start the RealSense pipeline using the configuration from the Config object
        pipeline.start(config_obj.pipeline_config)
        pipeline_started = True  # Mark the pipeline as started
        logger.info("Video pipeline started successfully.")

        # Video stream and recording setup
        frame_size = config_obj.config_dict['video_settings'].get('frameSize')  # Fallback to default size
        cv2.namedWindow("FHSBG Video Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FHSBG Video Stream", *frame_size)
        out = cv2.VideoWriter(
            config_obj.config_dict['video_settings']['filename'],
            config_obj.config_dict['video_settings']['fourcc'],
            config_obj.config_dict['video_settings']['fps'],
            frame_size
        )

        while True:
            # Wait for the next set of frames from the camera
            frames = await asyncio.to_thread(pipeline.wait_for_frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue  # If any frame is missing, skip the current iteration

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)  # Convert the image from BGR to HSV color space
            height, width, _ = color_image.shape
            center_x, center_y = width // 2, height // 2
            cv2.circle(color_image, (center_x, center_y), 5, (0,255,0), -1)
            # Threshold the HSV image to get only the colors defined in the configuration
            mask1 = cv2.inRange(hsv_image, config_obj.config_dict['color_range']['lower_red1'], config_obj.config_dict['color_range']['upper_red1'])
            mask2 = cv2.inRange(hsv_image, config_obj.config_dict['color_range']['lower_red2'], config_obj.config_dict['color_range']['upper_red2'])
            mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Update tracked objects
            current_time = time.time()
            valid_objects = []  # List to keep valid objects that meet the time condition
            for contour in contours:
                found = False
                for tracked_obj in tracked_objects:
                    # Update tracked object if it matches with the existing one
                    if cv2.matchShapes(tracked_obj.contour, contour, 1, 0.0) < 0.1:
                        tracked_obj.contour = contour  # Update contour
                        tracked_obj.timestamp = current_time  # Update timestamp
                        found = True
                        break
                if not found:
                    tracked_objects.append(TrackedObject(contour))  # Add new object
            # Remove old objects and draw blue squares on valid ones
            for obj in [t for t in tracked_objects if current_time - t.timestamp >= 3]:
                x, y, w, h = cv2.boundingRect(obj.contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue rectangle
                valid_objects.append(obj)  # Add to valid objects list

            # Drawing and processing for the largest object if it exists
            if valid_objects:
                largest_object = await find_largest_obstacle([obj.contour for obj in valid_objects])
                x, y, w, h = cv2.boundingRect(largest_object)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
                M = cv2.moments(largest_object)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)  # Mark the center of gravity

            # Display the resulting frame
            cv2.imshow("FHSBG Video Stream", color_image)
            out.write(color_image)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Sleep to yield execution to other tasks
            await asyncio.sleep(0.01)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if pipeline_started:
            pipeline.stop()  # Stop the pipeline if it was started
            logger.info("Video pipeline stopped.")
        out.release()  # Release the video writer
        cv2.destroyAllWindows()  # Close all OpenCV windows
        logger.info("Resources released.")

async def print_coordinates(video_processor_queue):
    try:
        while True:
            x_original, y_original, z_original, x_filtered, y_filtered, z_filtered = await video_processor_queue.get()
            print(f"Original: Dx = {x_original:.2f}, Dy = {y_original:.2f}, Dz = {z_original:.2f}")
            print(f"Filtered: Dx = {x_filtered:.2f}, Dy = {y_filtered:.2f}, Dz = {z_filtered:.2f}")
            print("-" * 45)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    video_processor_queue = asyncio.Queue()
    config = Config
    filter_config = LowPassFilter()

    try:
        # Create tasks within the event loop
        video_processor_task = loop.create_task(process_frames(video_processor_queue, config, filter_config))
        print_coordinates_task = loop.create_task(print_coordinates(video_processor_queue))
        # Run tasks until they complete or an exception is raised
        loop.run_until_complete(asyncio.gather(video_processor_task, print_coordinates_task))
    except KeyboardInterrupt:
        pass
