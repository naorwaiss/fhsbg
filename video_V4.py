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
        self.pipeline_config.enable_stream(rs.stream.color, 0, 1920, 1080, rs.format.bgr8, 30)

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

    pipeline.start(config_obj.pipeline_config)
    cv2.namedWindow("Color Stream", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    out = cv2.VideoWriter(config_obj.config_dict['video_settings']['filename'],
                          config_obj.config_dict['video_settings']['fourcc'],
                          config_obj.config_dict['video_settings']['fps'],
                          config_obj.config_dict['video_settings']['frameSize'])

    x_filter = LowPassFilter(filter_config.alpha)
    y_filter = LowPassFilter(filter_config.alpha)
    z_filter = LowPassFilter(filter_config.alpha)

    try:
        while True:
            frames = await asyncio.to_thread(pipeline.wait_for_frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Processing the frames
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())  # Apply necessary depth visualization conversion
            resized_color_image = cv2.resize(color_image, config_obj.config_dict['video_settings']['frameSize'])
            resized_depth_image = cv2.resize(depth_image, config_obj.config_dict['video_settings']['frameSize'])

            # Display the resulting frame
            cv2.imshow("Color Stream", resized_color_image)
            cv2.imshow("Depth Stream", resized_depth_image)

            out.write(resized_color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

async def print_coordinates(video_processor_queue):
    try:
        last_time = time.time()

        while True:
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time

            x_meters, y_meters, depth_value, x_filtered, y_filtered, z_filtered = await video_processor_queue.get()
            logger.info(f"Original: X={x_meters:.2f}, Y={y_meters:.2f}, Z={depth_value:.2f}")
            logger.info(f"Filtered: X={x_filtered:.2f}, Y={y_filtered:.2f}, Z={z_filtered:.2f}")
            logger.info("-" * 20)

    except asyncio.CancelledError:
        logger.info("Coordinate printing task cancelled.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    video_processor_queue = asyncio.Queue()
    config = Config
    filter_config = LowPassFilter()

    try:
        # Create tasks within the event loop
        frame_processor_task = loop.create_task(process_frames(video_processor_queue, config, filter_config))
        coordinate_printer_task = loop.create_task(print_coordinates(video_processor_queue))

        # Run tasks until they complete or an exception is raised
        loop.run_until_complete(asyncio.gather(frame_processor_task, coordinate_printer_task))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
