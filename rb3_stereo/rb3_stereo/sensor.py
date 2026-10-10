"""Low-level access to the two OV9282 global-shutter cameras on the RB3 Gen 2 vision mezzanine.

CAM0A: mainline ov9282 V4L2 driver (patched for 1 MIPI lane)
       -> msm_csiphy0 -> msm_csid0 -> msm_vfe0_rdi0 -> msm_vfe0_video0
CAM0B: no V4L2 sensor driver (powered by cam0b_bridge.ko), configured over I2C by cloning CAM0A's
       registers, captured through msm_csiphy1 = combo alias of CSIPHY0 (patched qcom-camss)
       -> msm_csid1 -> msm_vfe1_rdi0 -> msm_vfe1_video0

Frames are streamed with v4l2-ctl (OpenCV's V4L2 backend cannot open the multi-planar camss nodes):
image data on its stdout, per-frame kernel timestamps (CLOCK_MONOTONIC, end of frame) on stderr.
"""
import collections
import ctypes
import fcntl
import os
import re
import subprocess
import threading
import time

import numpy as np

CAM0B_CCI = "/sys/bus/platform/devices/ac4b000.cci"     # cci1, created by cam0b_bridge.ko
COMBO_PARAM = "/sys/module/qcom_camss/parameters/combo_phy"

# Every register the (patched) ov9282 driver writes, in its order, then control registers and
# PLL/MIPI registers the driver leaves at power-on defaults. A soft reset (0x0103) does not restore
# those - a leftover CamX init once made CAM0B run at 5/6 speed and corrupted its MIPI output.
# CAM0B gets the live values read from the running CAM0A.
CLONE_REGS = (
    0x0302, 0x030E, 0x3001, 0x3004, 0x3005, 0x3006, 0x3011, 0x3013, 0x301C, 0x3022, 0x3030, 0x3039,
    0x303A, 0x3503, 0x3505, 0x3507, 0x3508, 0x3610, 0x3611, 0x3620, 0x3632, 0x3633, 0x3666, 0x366F,
    0x3680, 0x3712, 0x372D, 0x3731, 0x3732, 0x377D, 0x3788, 0x3789, 0x378A, 0x378B, 0x3799, 0x3881,
    0x38A8, 0x38A9, 0x38B1, 0x38C4, 0x38C5, 0x38C6, 0x38C7, 0x3920, 0x4010, 0x4043, 0x4307, 0x4317,
    0x4501, 0x450A, 0x4601, 0x470F, 0x4F07, 0x5000, 0x5001, 0x5E00, 0x5D00, 0x5D01, 0x0101, 0x1000,
    0x5A08, 0x3778, 0x3800, 0x3801, 0x3802, 0x3803, 0x3804, 0x3805, 0x3806, 0x3807, 0x3808, 0x3809,
    0x380A, 0x380B, 0x3810, 0x3811, 0x3812, 0x3813, 0x3814, 0x3815, 0x3820, 0x3821, 0x4003, 0x4008,
    0x4009, 0x400C, 0x400D, 0x4507, 0x4509, 0x4837, 0x3500, 0x3501, 0x3502, 0x3509, 0x380C, 0x380D,
    0x380E, 0x380F, 0x3662,
    0x030A, 0x030B, 0x030C, 0x030D, 0x030F, 0x4800, 0x37A0, 0x3830, 0x3831,
)
EXPOSURE_REGS = (0x3500, 0x3501, 0x3502, 0x3509)        # exposure (3 bytes) + analogue gain
VTS_REG = 0x380E


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def check_setup():
    """Raise a helpful error unless the kernel side (patched camss + cam0b_bridge) is loaded."""
    try:
        combo = open(COMBO_PARAM).read().strip()
    except OSError:
        combo = None
    if combo != "0" or not os.path.isdir(CAM0B_CCI):
        raise RuntimeError("stereo kernel modules not loaded (patched qcom-camss with combo_phy=0 and "
                           "cam0b_bridge) - run: sudo rb3-stereo-setup")


def _cmdline(pid):
    return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()


def camera_users():
    """'pid command' of every other process holding a video, media or v4l-subdev device open."""
    users = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == os.getpid():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
            if not any(os.readlink(f"/proc/{pid}/fd/{fd}").startswith(("/dev/video", "/dev/media", "/dev/v4l-subdev"))
                       for fd in fds):
                continue
            cmd = _cmdline(pid)
            ppid = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1]
            if cmd.startswith("v4l2-ctl"):          # our capture helper: name the program that started it
                users.add(f"{ppid} {_cmdline(ppid)[:100]}")
            else:
                users.add(f"{pid} {cmd[:100]}")
        except OSError:
            continue        # not ours or gone
    return sorted(users)


def find_media():
    for i in range(30):
        for m in sorted(f"/dev/{d}" for d in os.listdir("/dev") if d.startswith("media")):
            r = subprocess.run(["media-ctl", "-d", m, "-p"], capture_output=True, text=True)
            if "camss" in r.stdout:
                return m
        time.sleep(0.2)
    raise RuntimeError("camss media device not found")


def cam0b_i2c_bus():
    buses = sorted(d for d in os.listdir(CAM0B_CCI) if d.startswith("i2c-"))
    if not buses:
        raise RuntimeError("cci1 has no I2C adapter (cam0b_bridge loaded?)")
    return int(buses[0].split("-")[1])                   # first master = CCI_I2C2 = CAM0B


class _I2CMsg(ctypes.Structure):
    _fields_ = [("addr", ctypes.c_uint16), ("flags", ctypes.c_uint16), ("len", ctypes.c_uint16),
                ("buf", ctypes.POINTER(ctypes.c_uint8))]


class _I2CRdwr(ctypes.Structure):
    _fields_ = [("msgs", ctypes.POINTER(_I2CMsg)), ("nmsgs", ctypes.c_uint32)]


class I2CDev:
    """16-bit-register I2C device via I2C_RDWR (one transaction, no subprocess, works next to a
    kernel driver bound to the same address)."""
    I2C_RDWR, I2C_M_RD = 0x0707, 0x0001

    def __init__(self, bus, addr=0x60):
        self.addr = addr
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        self.lock = threading.Lock()

    def _xfer(self, *msgs):
        arr = (_I2CMsg * len(msgs))(*msgs)
        with self.lock:
            fcntl.ioctl(self.fd, self.I2C_RDWR, _I2CRdwr(arr, len(msgs)))

    def write(self, reg, *vals):
        buf = (ctypes.c_uint8 * (2 + len(vals)))(reg >> 8, reg & 0xFF, *vals)
        self._xfer(_I2CMsg(self.addr, 0, len(buf), buf))

    def read(self, reg, n=1):
        wbuf = (ctypes.c_uint8 * 2)(reg >> 8, reg & 0xFF)
        rbuf = (ctypes.c_uint8 * n)()
        self._xfer(_I2CMsg(self.addr, 0, 2, wbuf), _I2CMsg(self.addr, self.I2C_M_RD, n, rbuf))
        return bytes(rbuf)

    def read8(self, reg):
        return self.read(reg)[0]

    def close(self):
        os.close(self.fd)


class Capture:
    """One camss video node streamed through v4l2-ctl. Keeps the newest frames with their index and
    the kernel frame-end timestamp of every frame (index = order of delivery)."""

    def __init__(self, video, width, height, ring=8):
        self.video, self.w, self.h = video, width, height
        run(["v4l2-ctl", "-d", video, f"--set-fmt-video=width={width},height={height},pixelformat=GREY"])
        info = run(["v4l2-ctl", "-d", video, "--get-fmt-video"])
        self.bpl = int(re.search(r"Bytes per Line\s*:\s*(\d+)", info).group(1))
        self.size = int(re.search(r"Size Image\s*:\s*(\d+)", info).group(1))
        self.frames = collections.deque(maxlen=ring)        # (index, HxW uint8)
        self.stamps = {}                                     # index -> CLOCK_MONOTONIC s
        self.recent = collections.deque(maxlen=1024)         # recent timestamps, for sync
        self.count = 0
        self._ts_count = 0
        self._cv = threading.Condition()
        self._proc = subprocess.Popen(
            ["v4l2-ctl", "-d", video, "--stream-mmap=4", "--stream-count=0", "--stream-to=-", "--verbose"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._read_frames, daemon=True).start()
        threading.Thread(target=self._read_stamps, daemon=True).start()

    def _read_frames(self):
        out = self._proc.stdout
        while True:
            buf = bytearray(self.size)
            mv, got = memoryview(buf), 0
            while got < self.size:
                n = out.readinto(mv[got:])
                if not n:
                    return
                got += n
            frame = np.frombuffer(buf, np.uint8)[: self.bpl * self.h].reshape(self.h, self.bpl)[:, : self.w]
            with self._cv:
                self.frames.append((self.count, frame))
                self.count += 1
                self._cv.notify_all()

    def _read_stamps(self):
        # big reads: reading the raw pipe line by line costs a syscall per byte and lagged ~0.5 s
        fd, rest = self._proc.stderr.fileno(), b""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                return
            *lines, rest = (rest + chunk).split(b"\n")
            for line in lines:
                m = re.search(rb"seq:\s*\d+.*?ts: ([\d.]+)", line)
                if m:
                    t = float(m.group(1))
                    with self._cv:
                        self.stamps[self._ts_count] = t
                        self.stamps.pop(self._ts_count - 256, None)
                        self.recent.append(t)
                        self._ts_count += 1
                        self._cv.notify_all()

    def stamp(self, index, timeout=0.1):
        with self._cv:
            self._cv.wait_for(lambda: index in self.stamps, timeout)
            return self.stamps.get(index)

    def wait_newer(self, index, timeout):
        """Wait until a frame with index > `index` exists; return the newest (index, frame) or None."""
        with self._cv:
            if not self._cv.wait_for(lambda: self.count - 1 > index, timeout):
                return None
            return self.frames[-1]

    def alive(self):
        return self._proc.poll() is None

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class Pipelines:
    """media-ctl setup of both capture paths."""

    A = ("msm_csiphy0", "msm_csid0", "msm_vfe0_rdi0", "msm_vfe0_video0")
    B = ("msm_csiphy1", "msm_csid1", "msm_vfe1_rdi0", "msm_vfe1_video0")

    def __init__(self):
        self.mdev = find_media()
        topo = run(["media-ctl", "-d", self.mdev, "-p"])
        self.sensor = re.search(r"entity \d+: (ov9282 [^ ]+) ", topo).group(1)
        self.subdev = self.entity(self.sensor)
        self.i2c_a = int(self.sensor.split()[1].split("-")[0])

    def entity(self, name):
        return run(["media-ctl", "-d", self.mdev, "-e", name]).strip()

    def mc(self, *a):
        run(["media-ctl", "-d", self.mdev, *a])

    def setup(self, width, height):
        fmt = f"Y8_1X8/{width}x{height}"
        try:
            self.mc("-r")
        except subprocess.CalledProcessError:
            users = camera_users()
            if users:
                raise RuntimeError("the cameras are in use by another process - stop it first "
                                   "(tmux attach -t <session>, Ctrl-C):\n  " + "\n  ".join(users)) from None
            raise
        self.mc("-V", f'"{self.sensor}":0[fmt:{fmt} field:none]')
        for csiphy, csid, rdi, _ in (self.A, self.B):
            self.mc("-l", f'"{csiphy}":1->"{csid}":0[1]')
            self.mc("-l", f'"{csid}":1->"{rdi}":0[1]')
            for e in (csiphy, csid):
                self.mc("-V", f'"{e}":0[fmt:{fmt}]')
                self.mc("-V", f'"{e}":1[fmt:{fmt}]')
            self.mc("-V", f'"{rdi}":0[fmt:{fmt}]')
        return self.entity(self.A[3]), self.entity(self.B[3])


class SensorA:
    """CAM0A through its V4L2 subdev controls (the ov9282 driver writes the registers)."""

    def __init__(self, subdev, i2c_bus):
        self.subdev = subdev
        self.i2c = I2CDev(i2c_bus)

    def ctrl(self, name, value=None):
        if value is not None:
            run(["v4l2-ctl", "-d", self.subdev, "--set-ctrl", f"{name}={int(value)}"])
        return int(run(["v4l2-ctl", "-d", self.subdev, "--get-ctrl", name]).split(":")[1])

    def ctrl_range(self, name):
        line = next(l for l in run(["v4l2-ctl", "-d", self.subdev, "--list-ctrls"]).splitlines()
                    if l.strip().startswith(name))
        return tuple(int(re.search(rf"{k}=(-?\d+)", line).group(1)) for k in ("min", "max", "default"))

    def wait_streaming(self, timeout=3.0):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            try:
                if self.i2c.read8(0x0100) == 1:
                    return
            except OSError:
                pass                                     # powered down until STREAMON
            time.sleep(0.005)
        raise RuntimeError("CAM0A did not start streaming")


class SensorB:
    """CAM0B over I2C only."""

    def __init__(self, i2c_bus):
        self.i2c = I2CDev(i2c_bus)

    def clone_from(self, a):
        """Soft reset, then copy every relevant register from the running CAM0A; leaves it in standby."""
        self.i2c.write(0x0103, 0x01)
        time.sleep(0.01)
        for r in CLONE_REGS:
            self.i2c.write(r, a.i2c.read8(r))

    def copy_regs(self, a, regs):
        for r in regs:
            self.i2c.write(r, a.i2c.read8(r))

    def stream(self, on):
        self.i2c.write(0x0100, 0x01 if on else 0x00)

    def vts(self):
        return int.from_bytes(self.i2c.read(VTS_REG, 2), "big")

    def set_vts(self, vts):
        self.i2c.write(VTS_REG, vts >> 8, vts & 0xFF)       # both bytes in one transaction
