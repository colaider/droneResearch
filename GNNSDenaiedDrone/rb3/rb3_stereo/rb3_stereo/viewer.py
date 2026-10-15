"""Remote viewer and control panel for the rb3_stereo camera node, from any machine on the same
ROS_DOMAIN_ID (default 42, as on the drone). Shows the JPEG preview by default (light enough for
Wi-Fi); --raw shows the full-rate raw topics (only on the drone itself or a wired link).

    ros2 run rb3_stereo viewer                          # or: python3 viewer.py (standalone file)
    ros2 run rb3_stereo viewer --fps 60 --exposure-us 3000 --gain 2
    ros2 run rb3_stereo viewer --auto-exposure on --ae-target 110 --set-only

Keys: q quit | s save | a auto exposure on/off | + - exposure | g h gain | 1 2 3 = 30/60/120 fps |
      r toggle 640x400 / 1280x800
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


def image_to_array(msg):
    ch = 3 if msg.encoding in ("bgr8", "rgb8") else 1
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, : msg.width * ch]
    return a.reshape(msg.height, msg.width, ch) if ch == 3 else a


class Viewer:
    def __init__(self, args):
        self.args = args
        ns = args.ns.rstrip("/")
        self.node = rclpy.create_node("stereo_viewer")
        self.params = AsyncParameterClient(self.node, f"{ns}/stereo_camera")
        self.status, self.img, self.raw = {}, None, {}
        self.status_t, self.sent, self.sent_t = 0.0, {}, 0.0      # status every 5 s; our own last changes
        self.lock = threading.Lock()
        n = self.node
        n.create_subscription(String, f"{ns}/status", self._on_status, 10)
        if args.raw:
            for side in ("left", "right"):
                n.create_subscription(Image, f"{ns}/{side}/image_raw", lambda m, s=side: self._on_raw(s, m),
                                      qos_profile_sensor_data)
        else:
            n.create_subscription(CompressedImage, f"{ns}/preview/compressed", self._on_preview,
                                  qos_profile_sensor_data)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(n)
        self.spinning = True
        self.spin_thread = threading.Thread(target=self._spin, daemon=True)
        self.spin_thread.start()
        self.frames, self.t_fps, self.rate = 0, time.monotonic(), 0.0

    def _spin(self):
        while self.spinning and rclpy.ok():
            self.ex.spin_once(timeout_sec=0.1)

    def close(self):
        self.spinning = False
        self.spin_thread.join(timeout=1.0)
        self.ex.shutdown()
        self.node.destroy_node()

    def _on_status(self, msg):
        self.status = json.loads(msg.data)
        self.status_t = time.monotonic()

    def _on_preview(self, msg):
        # as sent: the camera node's preview is mono, the VO runner's carries a coloured tracking overlay
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        with self.lock:
            self.img = img
            self.frames += 1

    def _on_raw(self, side, msg):
        with self.lock:
            self.raw[side] = (msg.header.stamp, image_to_array(msg))
            l, r = self.raw.get("left"), self.raw.get("right")
            if l and r and l[0] == r[0]:
                self.img = np.hstack((l[1], r[1]))
                self.frames += 1

    # ---------------------------------------------------------------- parameters

    def get(self, names):
        f = self.params.get_parameters(names)
        if not self._wait(f):
            return {}
        return {k: parameter_value_to_python(v) for k, v in zip(names, f.result().values)}

    def set(self, **kw):
        # atomically: e.g. resolution + fps are validated together
        f = self.params.set_parameters_atomically([Parameter(k, value=v) for k, v in kw.items()])
        if not self._wait(f, 15.0):
            print("set_parameters: no answer from the camera node")
            return
        r = f.result().result
        if r.successful:
            self.sent.update(kw)
            self.sent_t = time.monotonic()
        print(", ".join(f"{k}={v}" for k, v in kw.items()) + (": ok" if r.successful else f": REJECTED: {r.reason}"))

    @staticmethod
    def _wait(future, timeout=5.0):
        t_end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < t_end:
            time.sleep(0.01)
        return future.done()

    def apply_args(self):
        a, kw = self.args, {}
        if a.resolution:
            kw["resolution"] = a.resolution
        if a.fps:
            kw["fps"] = a.fps
        if a.auto_exposure:
            kw["auto_exposure"] = a.auto_exposure == "on"
        if a.exposure_us is not None or a.gain is not None:
            kw.setdefault("auto_exposure", False)
            if a.exposure_us is not None:
                kw["exposure_us"] = float(a.exposure_us)
            if a.gain is not None:
                kw["gain"] = float(a.gain)
        if a.ae_target is not None:
            kw["ae_target"] = float(a.ae_target)
        if a.ae_max_exposure_us is not None:
            kw["ae_max_exposure_us"] = float(a.ae_max_exposure_us)
        if a.preview_fps is not None:
            kw["preview_fps"] = float(a.preview_fps)
        if kw:
            self.set(**kw)

    # ---------------------------------------------------------------- UI

    def key(self, k):
        p = self.get(["auto_exposure", "exposure_us", "gain", "resolution"])
        if not p:
            print("camera node not reachable")
            return
        cur = dict(self.status)
        if self.sent_t > self.status_t:                 # repeated key presses: build on what we sent
            cur.update(self.sent)
        exp, gain = cur.get("exposure_us", p["exposure_us"]), cur.get("gain", p["gain"])
        if k == ord("a"):
            self.set(auto_exposure=not p["auto_exposure"])
        elif k in (ord("+"), ord("=")):
            self.set(auto_exposure=False, exposure_us=float(exp) * 1.25, gain=float(gain))
        elif k == ord("-"):
            self.set(auto_exposure=False, exposure_us=float(exp) / 1.25, gain=float(gain))
        elif k == ord("g"):
            self.set(auto_exposure=False, exposure_us=float(exp), gain=float(gain) * 1.25)
        elif k == ord("h"):
            self.set(auto_exposure=False, exposure_us=float(exp), gain=float(gain) / 1.25)
        elif k in (ord("1"), ord("2"), ord("3")):
            self.set(fps={ord("1"): 30, ord("2"): 60, ord("3"): 120}[k])
        elif k == ord("r"):
            res = "1280x800" if p["resolution"] == "640x400" else "640x400"
            self.set(resolution=res, **({"fps": 60} if res == "1280x800" else {}))

    def run(self):
        win = "rb3 stereo (q quit, s save, a AE, +/- exposure, g/h gain, 1/2/3 fps, r resolution)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        while True:
            with self.lock:
                img = None if self.img is None else self.img.copy()
            now = time.monotonic()
            if now - self.t_fps >= 1.0:
                self.rate, self.frames, self.t_fps = self.frames / (now - self.t_fps), 0, now
            if img is None:
                img = np.zeros((200, 640), np.uint8)
                cv2.putText(img, f"waiting for {self.args.ns} (ROS_DOMAIN_ID {self.args.domain})", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1)
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
            if self.args.scale != 1.0:
                disp = cv2.resize(disp, None, fx=self.args.scale, fy=self.args.scale)
            s = self.status
            lines = [f"camera {s.get('mode', '?')} {s.get('fps', 0):.1f} fps | shown {self.rate:.1f} fps",
                     f"exposure {s.get('exposure_us', '?')} us  gain {s.get('gain', '?')}  "
                     f"AE {'on' if s.get('auto_exposure') else 'off'}"
                     + (f"  mean {s['mean']}" if 'mean' in s else "")]
            if "sync_offset_us" in s:
                lines.append(f"L/R sync offset {s.get('sync_offset_us')} us")
            # anything else in the status (e.g. the VO test runner's velocity), a few fields per line
            known = {"mode", "fps", "exposure_us", "gain", "auto_exposure", "mean", "sync_offset_us"}
            extra = [f"{k} {v}" for k, v in s.items() if k not in known]
            lines += ["  ".join(extra[i:i + 4]) for i in range(0, len(extra), 4)]
            for i, t in enumerate(lines):
                cv2.putText(disp, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(win, disp)
            k = cv2.waitKey(15) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s") and img is not None:
                name = time.strftime("stereo_%Y%m%d_%H%M%S")
                if self.args.raw and "left" in self.raw:
                    cv2.imwrite(f"{name}_left.png", self.raw["left"][1])
                    cv2.imwrite(f"{name}_right.png", self.raw["right"][1])
                else:
                    cv2.imwrite(f"{name}_preview.png", img)
                print("saved", name)
            elif k != 255:
                threading.Thread(target=self.key, args=(k,), daemon=True).start()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="/stereo", help="camera namespace (default /stereo)")
    ap.add_argument("--domain", type=int, default=int(os.environ.get("ROS_DOMAIN_ID", 42)),
                    help="ROS_DOMAIN_ID (default: $ROS_DOMAIN_ID or 42)")
    ap.add_argument("--raw", action="store_true", help="show full-rate raw images instead of the preview")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale")
    ap.add_argument("--resolution", choices=("640x400", "1280x800"))
    ap.add_argument("--fps", type=int, choices=(30, 60, 120))
    ap.add_argument("--auto-exposure", choices=("on", "off"))
    ap.add_argument("--exposure-us", type=float, help="manual exposure (turns auto exposure off)")
    ap.add_argument("--gain", type=float, help="manual analogue gain 1.0-15.9 (turns auto exposure off)")
    ap.add_argument("--ae-target", type=float, help="auto exposure target brightness 0-255")
    ap.add_argument("--ae-max-exposure-us", type=float, help="auto exposure exposure limit")
    ap.add_argument("--preview-fps", type=float, help="rate of the preview topic (0 = off)")
    ap.add_argument("--set-only", action="store_true", help="apply the settings and exit (no window)")
    args = ap.parse_args()

    rclpy.init(domain_id=args.domain)
    v = Viewer(args)
    settings = any(getattr(args, k) is not None for k in (
        "resolution", "fps", "auto_exposure", "exposure_us", "gain", "ae_target", "ae_max_exposure_us",
        "preview_fps"))
    if settings or args.set_only:      # only wait for the camera node when there is something to set
        if not v.params.wait_for_services(timeout_sec=10.0):
            print(f"camera node {args.ns}/stereo_camera not found on ROS_DOMAIN_ID {args.domain}")
        else:
            v.apply_args()
    if not args.set_only:
        v.run()
    v.close()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
