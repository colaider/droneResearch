"""Live plot of accelerometer, gyro and barometer -- runs ON THE DRONE, shows over X11.

Subscribes to the topics start_rb3 publishes and draws three rolling plots with matplotlib.
Over an SSH session with X11 forwarding (MobaXterm does this automatically) the window opens
on your computer. No camera, no web server -- just the three plots.

On the drone (ROS 2 sourced, start_rb3 running, X11 forwarding on):
    python3 plot_live.py
    python3 plot_live.py --ns /drone --window 10

If plt.show() complains about no display/backend:  sudo apt install python3-tk
"""
import argparse
import collections
import threading

import matplotlib.pyplot as plt
import rclpy
from matplotlib.animation import FuncAnimation
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped

MAXLEN = 4000     # samples kept per trace


class SensorBuffers(Node):
    def __init__(self, ns="/drone"):
        super().__init__("plot_live")
        d = lambda: collections.deque(maxlen=MAXLEN)
        self.t = d()                         # imu timestamps
        self.ax, self.ay, self.az = d(), d(), d()
        self.gx, self.gy, self.gz = d(), d(), d()
        self.bt, self.baro = d(), d()        # barometer timestamps + altitude

        ns = ns.rstrip("/")
        self.create_subscription(Imu, f"{ns}/imu", self._on_imu, qos_profile_sensor_data)
        self.create_subscription(
            PointStamped,
            f"{ns}/baro_altitude_stamped",
            self._on_baro,
            qos_profile_sensor_data,
        )

    def _on_imu(self, m):
        self.t.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        self.ax.append(m.linear_acceleration.x)
        self.ay.append(m.linear_acceleration.y)
        self.az.append(m.linear_acceleration.z)
        self.gx.append(m.angular_velocity.x)
        self.gy.append(m.angular_velocity.y)
        self.gz.append(m.angular_velocity.z)

    def _on_baro(self, m):
        self.bt.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        self.baro.append(float(m.point.z))


def main():
    p = argparse.ArgumentParser(description="live accel / gyro / baro plot")
    p.add_argument("--ns", default="/drone", help="drone namespace (default /drone)")
    p.add_argument("--window", type=float, default=10.0, help="seconds shown on screen")
    args = p.parse_args()

    rclpy.init()
    node = SensorBuffers(args.ns)

    # spin ROS in the background so matplotlib owns the main thread
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    fig, (ax_a, ax_g, ax_b) = plt.subplots(3, 1, figsize=(9, 8))
    fig.suptitle("RB3 live sensors")

    la = [ax_a.plot([], [], label=lbl)[0] for lbl in ("x", "y", "z")]
    lg = [ax_g.plot([], [], label=lbl)[0] for lbl in ("x", "y", "z")]
    (lb,) = ax_b.plot([], [], color="tab:orange")

    ax_a.set_ylabel("accel  m/s²");  ax_a.legend(loc="upper right", ncol=3)
    ax_g.set_ylabel("gyro  rad/s");  ax_g.legend(loc="upper right", ncol=3)
    ax_b.set_ylabel("baro alt  m");  ax_b.set_xlabel("time  [s]")
    for ax in (ax_a, ax_g, ax_b):
        ax.grid(True, alpha=0.3)

    def rescale(ax, tmax):
        ax.set_xlim(max(0.0, tmax - args.window), max(args.window, tmax))
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)

    def update(_frame):
        t = list(node.t)
        if t:
            for line, buf in zip(la, (node.ax, node.ay, node.az)):
                line.set_data(t, list(buf))
            for line, buf in zip(lg, (node.gx, node.gy, node.gz)):
                line.set_data(t, list(buf))
            rescale(ax_a, t[-1])
            rescale(ax_g, t[-1])
        bt = list(node.bt)
        if bt:
            lb.set_data(bt, list(node.baro))
            rescale(ax_b, bt[-1])
        return (*la, *lg, lb)

    _ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
