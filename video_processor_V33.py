import asyncio
import cv2
import numpy as np
import pyrealsense2 as rs
import time

class TrackedObject:
    """Represents a tracked object in the scene."""
    def __init__(self, contour):
        self.contour = contour
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.is_persistent = False

    def update(self, new_contour):
        """Update the tracked object with new contour information."""
        self.contour = new_contour
        self.last_seen = time.time()

    def check_persistence(self, duration=3):
        """Check if the object has been present for a specified duration."""
        if (self.last_seen - self.first_seen) > duration and not self.is_persistent:
            self.is_persistent = True
            return True
        return False

def find_largest_contour(contours):
    """Finds the largest contour by area from a list of contours."""
    return max(contours, key=cv2.contourArea) if contours else None

async def process_frames():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    current_objects = []

    try:
        while True:
            frames = await asyncio.to_thread(pipeline.wait_for_frames)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

            # Red color detection
            red_mask = cv2.inRange(hsv_image, np.array([0, 120, 70]), np.array([10, 255, 255])) | \
                       cv2.inRange(hsv_image, np.array([160, 120, 70]), np.array([180, 255, 255]))

            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            updated_objects = []

            for contour in contours:
                if cv2.contourArea(contour) < 100:  # Filter out small objects
                    continue
                match_found = False
                for obj in current_objects:
                    if cv2.matchShapes(obj.contour, contour, 1, 0.0) < 0.5:
                        obj.update(contour)
                        updated_objects.append(obj)
                        match_found = True
                        break
                if not match_found:
                    updated_objects.append(TrackedObject(contour))

            # Drawing and persistence check
            for obj in updated_objects:
                x, y, w, h = cv2.boundingRect(obj.contour)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red frame for all detected objects
                if obj.check_persistence():
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue frame for persistent objects

            # Update the list with current frame's objects
            current_objects = updated_objects

            # Find the largest persistent object
            persistent_objects = [obj for obj in current_objects if obj.is_persistent]
            if persistent_objects:
                largest_obj = find_largest_contour([obj.contour for obj in persistent_objects])
                if largest_obj is not None:
                    x, y, w, h = cv2.boundingRect(largest_obj)
                    M = cv2.moments(largest_obj)
                    if M["m00"] != 0:  # Avoid division by zero
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green frame for the largest object
                        cv2.circle(color_image, (cX, cY), 5, (0, 255, 0), -1)  # Center of gravity

            cv2.imshow("Detected Red Objects", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(process_frames())
