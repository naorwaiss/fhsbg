import asyncio
from video_processor_V17 import Config, process_frames, print_coordinates

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    video_processor_queue = asyncio.Queue()
    config = Config
    try:
        # Create tasks within the event loop
        video_processor_task = loop.create_task(process_frames(video_processor_queue, config))
        print_coordinates_task = loop.create_task(print_coordinates(video_processor_queue))
        # Run tasks until they complete or an exception is raised
        loop.run_until_complete(asyncio.gather(video_processor_task, print_coordinates_task))
    except KeyboardInterrupt:
        pass