import asyncio
from video_processor import Config, process_frames, print_coordinates
from position_velocity import process_frames_with_custom_velocity, print_velocitys

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    video_processor_queue = asyncio.Queue()
    position_velocity_queue = asyncio.Queue()
    config = Config
    try:
        # Create tasks within the event loop
        video_processor_task = loop.create_task(process_frames(video_processor_queue, config))
        print_coordinates_task = loop.create_task(print_coordinates(video_processor_queue))
        position_velocity_task = loop.create_task(process_frames_with_custom_velocity(video_processor_queue, position_velocity_queue))
        print_velocity_task = loop.create_task(print_velocitys(position_velocity_queue))
        # Run tasks until they complete or an exception is raised
        loop.run_until_complete(asyncio.gather(video_processor_task, print_coordinates_task, position_velocity_task, print_velocity_task))
    except KeyboardInterrupt:
        pass