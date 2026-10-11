import sys
import time
import types
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from realCameraTest.realDroneStruct import RealDroneStruct
from swarm.droneClasses.dronePositionCTRL import DronePositionCTRL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DT = 0.01                                  
SETPOINT = np.array([0.0, 0.0, 0.5, 0.0])  
THROTTLE_CAP = 0.8                        

def main():
    rclpy.init()
    node = rclpy.create_node("control_loop")
    drone = RealDroneStruct(node=node)                       # shares this node
    pub = node.create_publisher(Twist, "/drone/cmd_attitude", 10)
    ctrl = DronePositionCTRL(drone, dt=DT)

    print("hovering at 0.5 m -- Ctrl-C to stop (radio takes over)", flush=True)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.0)           # pump the IMU/baro callbacks
        ctrl.step()                                      # integrate attitude/vel/pos from IMU
        roll, pitch, yaw_rate, throttle = ctrl.position_control(SETPOINT)
        throttle = float(np.clip(throttle, 0.0, THROTTLE_CAP))

        msg = Twist()
        msg.angular.x = float(roll)
        msg.angular.y = float(pitch)
        msg.angular.z = float(yaw_rate)
        msg.linear.z = throttle
        pub.publish(msg)
        
        time.sleep(DT)

    drone.close()


if __name__ == "__main__":
    main()
