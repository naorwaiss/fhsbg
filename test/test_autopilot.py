import asyncio
from autopilot_V13 import System, taking_off_prc, fly,landing_prc

if __name__ == "__main__":
    drone = System()
    loop = asyncio.get_event_loop()
    autopilot_queue = asyncio.Queue()
    try:
        # Create tasks within the event loop
        take_off_task = loop.create_task(taking_off_prc(drone))
        autopilot_task = loop.create_task(fly(drone, 1, 0, 0))
        landing_task = loop.create_task(landing_prc(drone))
        # Run tasks until they complete or an exception is raised
        loop.run_until_complete((take_off_task, autopilot_task, landing_task))
    except KeyboardInterrupt:
        pass