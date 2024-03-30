import asyncio
import cv2
import numpy as np
import pyrealsense2 as rs
import time

class TrackedObject:
    """Class representing objects being tracked."""
    def __init__(self, contour, first_seen):
        self.contour = contour
        self.first_seen = first_seen
        self.last_seen = first_seen
        self.is_persistent = False  # True if object has been in the image for more than 3 seconds

async def process_frames():
    """Processes video frames to detect and track red objects."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        current_objects = []

        while True:
            frames = await asyncio.to_thread(pipeline.wait_for_frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # Define range for red color and create masks
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Find contours for detected red objects
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_time = time.time()

            # Update or add new objects based on current visibility
            new_objects = []
            for contour in contours:
                if cv2.contourArea(contour) < 100:  # Filter out small contours
                    continue
                found = False
                for obj in current_objects:
                    if cv2.matchShapes(obj.contour, contour, 1, 0.0) < 0.5:
                        obj.contour = contour  # Update the contour
                        obj.last_seen = current_time  # Update last seen time
                        if current_time - obj.first_seen > 3:
                            obj.is_persistent = True  # Mark as persistent if visible for more than 3 seconds
                        found = True
                        break
                if not found:
                    new_objects.append(TrackedObject(contour, current_time))
            current_objects = new_objects + [obj for obj in current_objects if current_time - obj.last_seen < 1]  # Remove old objects

            # Process current objects for visualization
            largest_area = 0
            largest_persistent_obj = None
            for obj in current_objects:
                x, y, w, h = cv2.boundingRect(obj.contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw red frame
                if obj.is_persistent:
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw blue frame if persistent
                    area = cv2.contourArea(obj.contour)
                    if area > largest_area:
                        largest_area = area
                        largest_persistent_obj = obj

            # Highlight the largest persistent object
            if largest_persistent_obj:
                x, y, w, h = cv2.boundingRect(largest_persistent_obj.contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green frame for the largest object
                M = cv2.moments(largest_persistent_obj.contour)
                if M["m00"] != 0:  # Avoid division by zero
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(color_image, (cX, cY), 5, (0, 255, 0), -1)  # Center of gravity

            # Display the frame
            cv2.imshow("Detected Red Objects", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(process_frames())
