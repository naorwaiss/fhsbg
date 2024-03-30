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
        self.color_range = {
            'lower_red1': np.array([0, 120, 70]),
            'upper_red1': np.array([10, 255, 255]),
            'lower_red2': np.array([170, 120, 70]),
            'upper_red2': np.array([180, 255, 255]),
        }
        self.video_settings = {
            'filename': os.path.join(os.path.expanduser('~'), 'Desktop', 'output.avi'),
            'fourcc': cv2.VideoWriter_fourcc(*'MJPG'),
            'fps': 30.0,
            'frameSize': (640, 480)  # Set for display resolution
        }
        self.fov_horizontal = 69
        self.fov_vertical = 42
        self.pipeline_config = rs.config()
        self.set_pipeline_config()

    def set_pipeline_config(self):
        self.pipeline_config.enable_stream(rs.stream.depth, 0, 1280, 720, rs.format.z16, 30)
        self.pipeline_config.enable_stream(rs.stream.color, 0, 1280, 720, rs.format.bgr8, 30)

class TrackedObject:
    def __init__(self, contour, timestamp):
        self.contour = contour
        self.timestamp = timestamp
        self.is_valid = False  # Initially, object is not valid until it stays for more than 3 seconds

async def process_frames(config):
    pipeline = rs.pipeline()
    config_obj = config()

    try:
        # Start the RealSense pipeline using the provided configuration
        pipeline.start(config_obj.pipeline_config)
        logger.info("Video pipeline started successfully.")

        cv2.namedWindow("Real-time Tracking", cv2.WINDOW_AUTOSIZE)

        tracked_objects = []
        start_time = None

        while True:
            frames = await asyncio.to_thread(pipeline.wait_for_frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # Creating masks to detect red color
            mask1 = cv2.inRange(hsv_image, config_obj.color_range['lower_red1'], config_obj.color_range['upper_red1'])
            mask2 = cv2.inRange(hsv_image, config_obj.color_range['lower_red2'], config_obj.color_range['upper_red2'])
            mask = cv2.bitwise_or(mask1, mask2)

            # Find contours and track objects
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_time = time.time()
            if start_time is None:
                start_time = current_time

            for contour in contours:
                # Create new TrackedObject for every contour found
                new_object = True
                for obj in tracked_objects:
                    if cv2.contourArea(contour) > 100:  # Filter small contours
                        if cv2.matchShapes(obj.contour, contour, 1, 0.0) < 0.5:
                            obj.contour = contour  # Update existing object
                            new_object = False
                            if current_time - obj.timestamp > 3:
                                obj.is_valid = True  # Validate object after 3 seconds
                            break
                if new_object and cv2.contourArea(contour) > 100:
                    tracked_objects.append(TrackedObject(contour, current_time))

            # Filter out old objects
            tracked_objects = [obj for obj in tracked_objects if current_time - obj.timestamp < 10]  # Change as per your requirement

            # Draw rectangles around valid objects
            for obj in tracked_objects:
                if obj.is_valid:
                    x, y, w, h = cv2.boundingRect(obj.contour)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle for valid objects

            # Find and highlight the largest valid object
            if tracked_objects:
                valid_objects = [obj for obj in tracked_objects if obj.is_valid]
                if valid_objects:
                    largest_obj = max(valid_objects, key=lambda x: cv2.contourArea(x.contour))
                    x, y, w, h = cv2.boundingRect(largest_obj.contour)
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle for the largest object

            cv2.imshow("Real-time Tracking", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        logger.info("Video pipeline stopped and resources released.")

if __name__ == "__main__":
    asyncio.run(process_frames(Config))
