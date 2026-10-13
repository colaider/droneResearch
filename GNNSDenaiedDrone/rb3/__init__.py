"""Simple RB3 topic classes for the GNSS-denied drone.

One small class per topic, each reusing the proven drivers in rb3_control / rb3_stereo:
  ImuTopic      -> publishes the RB3's own IMU (gyro + accel, sensor core)   <ns>/imu
  BaroTopic     -> publishes the RB3 barometer                               <ns>/pressure, <ns>/baro_altitude
  CameraTopic   -> publishes the OV9282 stereo pair                          <ns>/left|right/image_raw (+ camera_info)
  CommandTopic  -> receives roll/pitch/yaw/throttle and commands the motors  <ns>/cmd_attitude -> Betaflight (MSP)

start_rb3.py starts all of them on one node.
"""
