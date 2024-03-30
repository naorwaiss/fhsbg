import asyncio
import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import os
import time

class Config:
    def __init__(self):
        self.camera_type = '435'
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
                'frameSize': (640, 480)
            }
        }

        self.pipeline_config = rs.config()
        self.set_pipeline_config()

    def set_pipeline_config(self):
        self.pipeline_config.enable_stream(rs.stream.depth, 0, *self.config_dict['video_settings']['frameSize'],
                                           rs.format.z16, int(self.config_dict['video_settings']['fps']))
        self.pipeline_config.enable_stream(rs.stream.color, 0, *self.config_dict['video_settings']['frameSize'],
                                           rs.format.bgr8, int(self.config_dict['video_settings']['fps']))

    def set_values(self):
        if self.camera_type == '435':
            self.fov_horizontal = 69
            self.fov_vertical = 42
        elif self.camera_type == '515':
            self.fov_horizontal = 70
            self.fov_vertical = 43
        else:
            raise ValueError("Invalid camera type. Supported types: '435', '515'.")
class LowPassFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.filtered_value = None

    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
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
    cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Stream", *config_obj.config_dict['video_settings']['frameSize'])
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
            color_image = np.asanyarray(color_frame.get_data())
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            height, width, _ = color_image.shape
            center_x, center_y = width // 2, height // 2
            cv2.circle(color_image, (center_x, center_y), 5, (255, 0, 0), -1)
            lower_red1 = config_obj.config_dict['color_range']['lower_red1']
            upper_red1 = config_obj.config_dict['color_range']['upper_red1']
            lower_red2 = config_obj.config_dict['color_range']['lower_red2']
            upper_red2 = config_obj.config_dict['color_range']['upper_red2']
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            largest_obstacle = await find_largest_obstacle(contours)
            if largest_obstacle is not None:
                M = cv2.moments(largest_obstacle)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    depth_value = depth_frame.get_distance(center_x, center_y)
                    adjusted_x = center_x - width // 2
                    adjusted_y = height // 2 - center_y
                    cv2.drawContours(color_image, [largest_obstacle], -1, (0, 0, 255), 2)
                    cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), -1)
                    x_meters, y_meters = await pixel_to_meters(adjusted_x, adjusted_y,
                                                               config_obj.fov_horizontal, config_obj.fov_vertical,
                                                               width, height, depth_value)
                    x_filtered = x_filter.update(x_meters)
                    y_filtered = y_filter.update(y_meters)
                    z_filtered = z_filter.update(depth_value)
                    await video_processor_queue.put((x_meters, y_meters, depth_value, x_filtered, y_filtered, z_filtered))
                else:
                    await video_processor_queue.put((0, 0, 0, 0, 0, 0))
            else:
                await video_processor_queue.put((0, 0, 0, 0, 0, 0))

            # Display the resulting frame
            cv2.imshow("Video Stream", color_image)
            out.write(color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"An error occurred: {e}")

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
            print(f"Original: X={x_meters:.2f}, Y={y_meters:.2f}, Z={depth_value:.2f}")
            print(f"Filtered: X={x_filtered:.2f}, Y={y_filtered:.2f}, Z={z_filtered:.2f}")
            print("-" * 20)

    except asyncio.CancelledError:
        pass

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
        pass
