"""
cd FHSBG_Code/
cd PX4-Autopilot/
make px4_sitl_default jmavsim
"""

import asyncio
from mavsdk import telemetry
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
log_handler = RotatingFileHandler('autopilot.log', maxBytes=10*1024*1024, backupCount=5)
log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('autopilot')
logger.setLevel(logging.INFO)
logger.propagate = False
logger.addHandler(log_handler)
logger.info("-----------------------------------------")

async def connect_drone(drone, system_address="udp://:14540", retries=3, delay=5):
    attempt = 0
    while attempt < retries:
        message = (f"Connect Drone:\n"
                     f"                  Attempt {attempt + 1}/{retries}")
        logger.info(message)
        print(message)
        try:
            await drone.connect(system_address=system_address)
            message = (f"                  Waiting for drone to connect at {system_address}...")
            logger.info(message)
            print(message)
            connection_task = asyncio.create_task(drone.core.connection_state().__anext__())
            state = await asyncio.wait_for(connection_task, delay)
            if state.is_connected:
                message = "                  Connected to drone!"
                logger.info(message)
                print(message)
                # Fetch and log system identification information
                identification = await drone.info.get_identification()
                message = (f"                  Drone Identification: Hardware UID - {identification.hardware_uid}\n"
                           f"                                        Legacy UID   - {identification.legacy_uid}")
                logger.info(message)
                print(message)
                message = "                  Subscribing to odometry data..."
                logger.info(message)
                print(message)
                async for odometry in drone.telemetry.odometry():
                    message = (f"                  Odometry data received!\n"
                               f"                  Time Stamp: {odometry.time_usec}")
                    logger.info(message)
                    print(message)
                    break
                return True
            else:
                message = "-- warning: Connection state indicates not connected."
                logger.warning(message)
                print(message)
        except asyncio.TimeoutError:
            message = "-- warning: Connection to drone timed out."
            logger.warning(message)
            print(message)
        except Exception as e:
            message = f"-- error: Failed during connection or information retrieval: {e}, Type: {e.__class__.__name__}"
            logger.error(message)
            print(message)
        finally:
            if attempt + 1 < retries:
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        attempt += 1
    return False

async def arm_drone(drone):
    """
    Asynchronously arms the drone, ensuring it's safe to fly, and provides feedback from the flight controller.
    Args:
        drone: An instance of a drone control object that supports async operations.
    Returns:
        bool: True if the drone was successfully armed within the timeout period, False otherwise.
    Raises:
        Exception: Propagates any unexpected exceptions from the drone control library.
    """
    message = ("Arming:\n                  "
               "Attempting to arm the drone...")
    logger.info(message)
    print(message)
    try:
        await drone.action.arm()
        message = "                  Drone armed successfully!"
        logger.info(message)
        print(message)
        timeout = 10  # seconds
        start_time = asyncio.get_event_loop().time()
        async for is_armed in drone.telemetry.armed():
            if is_armed:
                message = ("                  Confirmation: Drone is armed")
                logger.info(message)
                print(message)
                return True
            elif (asyncio.get_event_loop().time() - start_time) > timeout:
                message = "-- warning.Timeout: Drone did not arm within the expected time"
                logger.warning(message)
                print(message)
                return False
            await asyncio.sleep(0.1)  # Brief pause to yield control
    except Exception as e:
        message = f"-- error: Failed to arm the drone: {e}"
        logger.error(message, exc_info=True)
        print(message)
        return False
    # Safety check in case of unexpected flow
    message = "-- warning: Unexpected condition: Reached end of arm_drone function without arming confirmation."
    logger.warning(message)
    print(message)
    return False

async def disarm_drone(drone):
    """
    Asynchronously disarms the drone, ensuring it's safe to handle, and provides feedback from the flight controller.
    Args:
        drone: An instance of a drone control object that supports async operations.
    Returns:
        bool: True if the drone was successfully disarmed, False otherwise.
    Raises:
        Exception: Propagates any unexpected exceptions from the drone control library.
    """
    message = ("Disarming:\n                  "
               "Attempting to disarm the drone")
    logger.info(message)
    print(message)
    try:
        await drone.action.disarm()
        message = "                  Drone disarmed successfully"
        logger.info(message)
        print(message)
        # Optionally, confirm the drone is disarmed by checking the armed state
        timeout = 10  # seconds
        start_time = asyncio.get_event_loop().time()
        async for is_armed in drone.telemetry.armed():
            if not is_armed:
                message = ("                  Confirmation: Drone is disarmed")
                logger.info(message)
                print(message)
                return True
            elif (asyncio.get_event_loop().time() - start_time) > timeout:
                message = "-- warning.Timeout: Drone did not disarm within the expected time"
                logger.warning(message)
                print(message)
                return False
            await asyncio.sleep(0.1)  # Brief pause to yield control
    except Exception as e:
        message = f"-- error: Failed to disarm the drone: {e}"
        logger.error(message, exc_info=True)
        print(message)
        return False
    # Safety check in case of unexpected flow
    message = "-- warning: Unexpected condition: Reached end of disarm_drone function without disarming confirmation."
    logger.warning(message)
    print(message)
    return False

async def set_initial_setpoint(drone):
    """
    Asynchronously sets the initial setpoint for the drone and checks if the setpoint has been acknowledged
    by verifying the drone's velocity is as expected.
    Args:
        drone: An instance of a drone control object that supports async operations, specifically for offboard control.
    Returns:
        bool: True if the initial setpoint was successfully set and confirmed, False otherwise.
    Raises:
        Exception: Propagates specific exceptions related to setting the setpoint, if known and applicable.
    """
    message = ("Initial Setpoint:\n"
               "                  Setting initial setpoint...")
    logger.info(message)
    print(message)
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        message = ("                  Initial setpoint set!\n"
                   "                  verifying acknowledgment...")
        logger.info(message)
        print(message)
        if await confirm_setpoint_applied(drone):
            message = ("                  Confirm Setpoint Applied: Setpoint acknowledged and applied successfully!")
            logger.info(message)
            print(message)
            return True
        else:
            message = "-- warning: Failed to confirm setpoint application"
            logger.warning(message)
            print(message)
            await drone.action.hold()
            return False
    except Exception as e:
        message = f"-- error: Failed to set initial setpoint: {e}"
        logger.error(message, exc_info=True)
        print(message)
        await drone.action.hold()
        return False

async def confirm_setpoint_applied(drone: System, tolerance: float = 0.1):
    """
    Checks if the drone's velocity in all axes is close to zero (or a specified setpoint) within a given tolerance.
    Args:
        drone: The drone control object.
        tolerance: The tolerance within which the velocity is considered close to the setpoint (default 0.1 m/s).
    Returns:
        bool: True if the drone's velocity is within the specified tolerance, False otherwise.
    """
    async for odometry in drone.telemetry.odometry():
            velocity = odometry.velocity_body
            message = (f"                  Confirm Setpoint: VB_x: {velocity.x_m_s}\n"
                       f"                                    VB_y: {velocity.y_m_s}\n"
                       f"                                    VB_z: {velocity.z_m_s}")
            logger.info(message)
            print(message)
            if (abs(velocity.x_m_s) - 0.0) <= tolerance and (abs(velocity.y_m_s) - 0.0) <= tolerance and (abs(velocity.z_m_s) - 0.0) <= tolerance:
                message = "                                    Drone's velocity is confirm applied!"
                logger.info(message)
                print(message)
                return True
            else:
                await asyncio.sleep(0.1)  # Check velocity at the next odometry update.
                message = "                                    Drone's velocity is Check at the next odometry update"
                logger.info(message)
                print(message)
    message = "                                    Drone's velocity is not applied"
    logger.info(message)
    print(message)
    # If the function exits the loop without returning True, then it never confirmed the velocity was within tolerance.
    return False

async def enable_offboard_mode(drone):
    """Enables offboard mode with error handling."""
    message = ("Enable Offboard:\n"
               "                  Enabling Offboard Mode...")
    logger.info(message)
    print(message)
    try:
        await drone.offboard.start()
        message = ("                  Offboard Mode is enabled!")
        logger.info(message)
        print(message)
    except OffboardError as error:
        message = f"-- error: Starting offboard mode failed with error: {error}"
        logger.error(message)
        print(message)
        # Ensure that stopping offboard mode and disarming are attempted even in case of failure.
        try:
            await drone.offboard.stop()
            message = "                  Stopping Offboard Mode due to error"
            logger.info(message)
            print(message)
        except Exception as stop_error:
            message = f"-- error: Failed to stop offboard mode: {stop_error}"
            logger.error(message)
            print(message)
        try:
            await drone.action.disarm()
            message = "                  Disarming due to offboard mode error"
            logger.info(message)
            print(message)
        except Exception as disarm_error:
            message = f"-- error: Failed to disarm the drone: {disarm_error}"
            logger.error(message)
            print(message)
        return False
    return True

async def stop_offboard_mode(drone):
    """Stops offboard mode with comprehensive error handling and operational feedback."""
    message = ("Stop Offboard:\n"
               "                  Stopping Offboard Mode...")
    logger.info(message)
    print(message)
    try:
        await drone.offboard.stop()
        message = "                  Offboard Mode Successfully Stopped!"
        logger.info(message)
        print(message)
    except OffboardError as error:
        # Improved error feedback with more detailed message.
        message = f"-- error: Stopping offboard mode failed with error: {error}"
        logger.error(message)
        print(message)
        # Optionally, disarm the drone for safety, depending on your use case.
        try:
            message = "                  Attempting to Disarm the Drone for Safety"
            logger.info(message)
            print(message)
            await drone.action.disarm()
            message = "                  Drone Successfully Disarmed"
            logger.info(message)
            print(message)
        except Exception as disarm_error:
            message = f"-- error: Failed to disarm the drone: {disarm_error}"
            logger.error(message)
            print(message)
        # Raising the error to allow the calling context to handle or log the failure.
        raise
    except Exception as general_error:
        # Catch-all for any other exceptions not explicitly handled above.
        message = f"-- error: An unexpected error occurred while stopping offboard mode: {general_error}"
        logger.error(message)
        print(message)
        # Consider whether to disarm the drone in this case as well.
        raise

async def take_off_to_altitude(drone: System, target_altitude: float = 2.0):
    """
    Asynchronously commands the drone to take off to a specified altitude by continuously sending upward velocity commands
    until the target altitude is reached, providing detailed feedback and robust error handling.
    Args:
        drone: An instance of the MAVSDK System class, representing the drone.
        target_altitude: The target altitude in meters that the drone should reach.
    Returns:
        bool: True if the drone successfully reaches the target altitude, False otherwise.
    """
    message = (f"Take Off:\n"
               f"                  Initiating takeoff to {target_altitude} meters")
    logger.info(message)
    print(message)
    body_velocity = VelocityBodyYawspeed(0.0, 0.0, -1.0, 0.0)
    await drone.offboard.set_velocity_body(body_velocity)
    if not await confirm_airborne(drone): # Confirm the drone is airborne before continuing
        # If the drone is not airborne, stop the upward motion and return False
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        message = "-- error: Unable to confirm the drone is airborne. Takeoff aborted."
        logger.error(message)
        print(message)
        return False
    # Once airborne, continue to target altitude
    while True:
        if await confirm_altitude(drone, target_altitude):
            message = f"                  Drone climbing to {target_altitude} meters."
            logger.info(message)
            print(message)
            break
        else:
            await drone.offboard.set_velocity_body(body_velocity)            # Continuously send upward velocity until the target altitude is reached
            await asyncio.sleep(0.5)  # Wait a bit before checking altitude again
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)) # Stop the drone's upward motion by setting velocity to zero
    message = "                  Velocity command sent to stop the drone"
    logger.info(message)
    print(message)
    return True

async def confirm_airborne(drone: System):
    """Wait for the drone to be confirmed as airborne."""
    async for state in drone.telemetry.landed_state():
        if state == telemetry.LandedState.IN_AIR:
            message = "                  Airborne: Drone is airborne!"
            logger.info(message)
            print(message)
            return True
        await asyncio.sleep(1)
    message = "-- warning: Drone is still on the ground."
    logger.warning(message)
    print(message)
    return False

async def confirm_altitude(drone, target_altitude: float, tolerance: float = 0.01):
    """
    Checks if the drone reaches the target altitude above the takeoff point within a specified tolerance.
    Args:
        drone: The MAVSDK System instance for the drone.
        target_altitude: The target altitude in meters.
        tolerance: The acceptable range (in meters) from the target altitude.
    Returns:
        bool: True if the drone reaches the target altitude within tolerance, False otherwise.
    """
    while True:
        current_altitude = await drone.telemetry.position_velocity_ned().__anext__()
        drone_altitude = abs(current_altitude.position.down_m)
        message = (f"                  Altitude: target altitude: {target_altitude} meters.\n"
                   f"                            drone altitude:  {drone_altitude} meters.")
        logger.info(message)
        print(message)
        delta_altitude = target_altitude - drone_altitude
        if delta_altitude <= tolerance:
            message = f"                            Reached target altitude: {delta_altitude} meters within tolerance."
            logger.info(message)
            print(message)
            return True
        await asyncio.sleep(0.1)  # Throttle checking rate to every 1 second

async def control_velocity(drone, V_x: float = 0.0, V_y: float = 0.0, V_z: float = 0.0, Yaw_deg: float = 0.0):
    """
    Commands the drone to move with specified velocities.
    Args:
        drone: An instance of the drone control object.
        V_x: Forward velocity in m/s.
        V_y: Rightward velocity in m/s.
        V_z: Downward velocity in m/s.
        Yaw_deg: Yaw rotation speed in degrees/s.
    """
    try:
        message = ("control_velocity:")
        logger.info(message)
        print(message)
        body_velocity = VelocityBodyYawspeed(V_x, V_y, V_z, Yaw_deg)  # Set correct velocity values
        await drone.offboard.set_velocity_body(body_velocity)
        message = "                  Velocity command for flight sent"
        logger.info(message)
        print(message)
        await asyncio.sleep(0.1)
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))  # Stop the drone
        message = "                  Velocity command for stopping the drone sent"
        logger.info(message)
        print(message)
    except asyncio.CancelledError:
        message = "-- warning: Flight operation cancelled"
        logger.warning(message)
        print(message)
    except Exception as e:
        message = f"-- error: Failed to control velocity: {str(e)}"
        logger.error(message, exc_info=True)
        print(message)

async def land_drone(drone: System):
    """
    Lands the drone with error handling, status feedback, and waits for landing confirmation using an optimized approach.
    Args:
        drone: An instance of the MAVSDK System class, representing the drone.
    Returns:
        bool: True if the drone successfully lands, False otherwise.
    """
    message = ("Land:\n"
               "                  Initiating Landing Sequence")
    logger.info(message)
    print(message)
    try:
        await drone.action.land()
        message = "                  Landing Command Sent, Awaiting Confirmation"
        logger.info(message)
        print(message)
        if await wait_for_landing(drone):
            message = "                  Drone Has Successfully Landed"
            logger.info(message)
            print(message)
            return True
        else:
            message = "-- warning: Timeout or Error in Confirming Landing"
            logger.warning(message)
            print(message)
            return False
    except Exception as e:
        message = "-- error: Failed to land the drone: {e}"
        logger.error(message, exc_info=True)
        print(message)
        return False

async def wait_for_landing(drone: System, timeout: int = 60):
    """
    Waits for the drone to confirm landing status within a given timeout.
    Args:
        drone: The drone control system.
        timeout: Timeout in seconds to wait for landing confirmation.
    Returns:
        bool: True if landed confirmation received, False otherwise.
    """
    end_time = asyncio.get_event_loop().time() + timeout
    async for state in drone.telemetry.landed_state():
        if state == telemetry.LandedState.ON_GROUND:
            return True
        if asyncio.get_event_loop().time() > end_time:
            break
        await asyncio.sleep(1)  # Efficient sleep, prevents tight looping
    return False

async def taking_off_prc(drone):
    try:
        await connect_drone(drone)
        await arm_drone(drone)
        await set_initial_setpoint(drone)
        await enable_offboard_mode(drone)
        await take_off_to_altitude(drone, 3)
    except Exception as e:
        drone.logger.error(f"An error occurred: {e}")
    await control_velocity(drone, V_x=0.0, V_y=0.0, V_z=-0.0)
    await asyncio.sleep(3)

async def landing_prc(drone):
    await control_velocity(drone, V_x=0.0, V_y=0.0, V_z=-0.0)
    await asyncio.sleep(3)
    try:
        await land_drone(drone)
        await stop_offboard_mode(drone)
        await disarm_drone(drone)
    except Exception as e:
        drone.logger.error(f"An error occurred: {e}")
async def run():
    drone = System()
    try:
        await taking_off_prc(drone)
        await control_velocity (drone, V_y=2.0)
        await control_velocity(drone, V_y=-2.0)
        await landing_prc(drone)
    except Exception as e:
        logger.error(f"An error occurred during the run sequence: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(run())