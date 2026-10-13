"""Simple RB3 topic classes for the GNSS-denied drone.

One small class per topic, each reusing the proven drivers in rb3_control / rb3_stereo:
  ImuTopic      -> publishes the RB3's own IMU (gyro + accel, sensor core)   <ns>/imu
  BaroTopic     -> publishes the RB3 barometer                               <ns>/pressure, <ns>/baro_altitude
  CameraTopic   -> publishes the OV9282 stereo pair                          <ns>/left|right/image_raw (+ camera_info)
  CommandTopic  -> receives roll/pitch/yaw/throttle and commands the motors  <ns>/cmd_attitude -> Betaflight (MSP)

start_rb3.py starts all of them on one node.
"""
import os as _os
import sys as _sys

# rb3_control and rb3_stereo are nested under this folder; put their package roots on sys.path
# so `import rb3_control` / `import rb3_stereo` resolve to the moved copies.
_HERE = _os.path.dirname(_os.path.abspath(__file__))
for _pkg in ("rb3_control", "rb3_stereo"):
    _root = _os.path.join(_HERE, _pkg)
    if _os.path.isdir(_root) and _root not in _sys.path:
        _sys.path.insert(0, _root)
