"""MSP (MultiWii Serial Protocol) client for Betaflight over USB CDC (/dev/ttyACM*).

Pure Python, no ROS dependency, so it can be used from plain scripts and tests.
Covers what the controller needs:
  - request/response for telemetry and configuration
  - MSP_SET_RAW_RC stick streaming from a background thread
  - discovery of modes (ARM, MSP OVERRIDE, ...) and the RC channel map
Frames are MSP v1, including jumbo frames (payload > 254 bytes, used by MSP_BOXNAMES).
"""
from __future__ import annotations

import math
import struct
import threading
import time

import serial

# --- MSP command ids -------------------------------------------------------
API_VERSION = 1
FC_VARIANT = 2
FC_VERSION = 3
BOARD_INFO = 4
MODE_RANGES = 34
FEATURE_CONFIG = 36
CF_SERIAL_CONFIG = 54
RX_MAP = 64
RAW_IMU = 102
MOTOR = 104
RC = 105
ATTITUDE = 108
ALTITUDE = 109
ANALOG = 110
BOXNAMES = 116
BOXIDS = 119
STATUS_EX = 150
SET_RAW_RC = 200

# --- Betaflight constants --------------------------------------------------
BOX_ARM = 0                     # permanent box ids
BOX_ANGLE = 1
BOX_MSP_OVERRIDE = 50
FUNCTION_MSP = 1 << 0
CHANNEL_COUNT = 16              # channels sent in MSP_SET_RAW_RC
ROLL, PITCH, YAW, THROTTLE = 0, 1, 2, 3     # internal (post-rcmap) channel order
AUX1 = 4
ACC_LSB_PER_G = 512             # MSP_RAW_IMU accelerometer scaling
ARMING_FAILSAFE = 1 << 1        # arming-disable bits meaning "radio link lost"
ARMING_RXLOSS = 1 << 2

# Arming disable flag names, bits 0-24 (stable from 4.2 to 4.6). Newer firmware appends
# flags after bit 24; ARMSWITCH is always the last (bit = reported flag count - 1).
ARMING_FLAG_NAMES = [
    "NOGYRO", "FAILSAFE", "RXLOSS", "BADRX", "BOXFAILSAFE", "RUNAWAY", "CRASH",
    "THROTTLE", "ANGLE", "BOOTGRACE", "NOPREARM", "LOAD", "CALIB", "CLI", "CMS",
    "BST", "MSP", "PARALYZE", "GPS", "RESCUE_SW", "RPMFILTER", "REBOOT_REQD",
    "DSHOT_BBANG", "NO_ACC_CAL", "MOTOR_PROTO",
]


class MspError(RuntimeError):
    pass


class Msp:
    """Blocking MSP client. One instance owns one serial port."""

    def __init__(self, port="/dev/ttyACM0", baud=115200, timeout=0.01):
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.port = port
        self._buf = bytearray()
        self._write_lock = threading.Lock()

    # --- framing ----------------------------------------------------------
    def send(self, cmd, payload=b""):
        body = bytes([len(payload), cmd]) + payload
        checksum = 0
        for b in body:
            checksum ^= b
        with self._write_lock:
            self.ser.write(b"$M<" + body + bytes([checksum]))

    def request(self, cmd, payload=b"", timeout=0.5):
        """Send a command and return its response payload, or None on timeout."""
        self.send(cmd, payload)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._buf += self.ser.read(512)
            frame = self._pop_frame()
            while frame is not None:
                kind, frame_cmd, data = frame
                if frame_cmd == cmd:
                    if kind == "!":
                        raise MspError("flight controller rejected MSP command %d" % cmd)
                    return data
                frame = self._pop_frame()        # skip acks of streamed commands
        return None

    def _pop_frame(self):
        while True:
            start = self._buf.find(b"$M")
            if start < 0:
                del self._buf[:-1]
                return None
            del self._buf[:start]
            if len(self._buf) < 5:
                return None
            kind = chr(self._buf[2])
            if kind not in ">!":
                del self._buf[:2]
                continue
            size, cmd = self._buf[3], self._buf[4]
            header = 5
            if size == 255:                      # jumbo frame: 16-bit length after the command
                if len(self._buf) < 7:
                    return None
                size = struct.unpack_from("<H", self._buf, 5)[0]
                header = 7
            if len(self._buf) < header + size + 1:
                return None
            data = bytes(self._buf[header:header + size])
            checksum = 0
            for b in self._buf[3:header + size]:
                checksum ^= b
            valid = checksum == self._buf[header + size]
            del self._buf[:header + size + 1]
            if valid:
                return kind, cmd, data

    def close(self):
        try:
            self.ser.close()
        except serial.SerialException:
            pass

    # --- identity / configuration ----------------------------------------
    def identity(self):
        api = self.request(API_VERSION, timeout=1.0)
        if api is None:
            return None
        variant = self.request(FC_VARIANT) or b"????"
        version = self.request(FC_VERSION) or b"\0\0\0"
        board = self.request(BOARD_INFO) or b""
        return {
            "api": "%d.%d" % (api[1], api[2]),
            "variant": variant.decode(errors="replace"),
            "version": "%d.%d.%d" % (version[0], version[1], version[2]),
            "board": board[:4].decode(errors="replace") if len(board) >= 4 else "?",
        }

    def serial_ports(self):
        """{port identifier: function mask}. Identifier 20 is the USB VCP."""
        data = self.request(CF_SERIAL_CONFIG) or b""
        ports = {}
        for i in range(0, len(data) - 6, 7):     # id u8, functions u16, 4 baud indexes u8
            identifier, function = struct.unpack_from("<BH", data, i)
            ports[identifier] = function
        return ports

    def rc_map(self):
        """rcmap[internal channel] = index in the MSP_SET_RAW_RC payload (TAER, AETR, ...)."""
        return list(self.request(RX_MAP) or range(8))

    def boxes(self):
        """[(permanent id, flag bit, name)] for every mode the firmware exposes."""
        names = self.request(BOXNAMES, timeout=1.0)
        ids = list(self.request(BOXIDS, timeout=1.0) or b"")
        if not names or not ids:
            return []
        labels = [n for n in names.decode(errors="replace").split(";") if n]
        return [(ids[i], 1 << i, label) for i, label in enumerate(labels) if i < len(ids)]

    def box_bit(self, permanent_id):
        for pid, bit, _name in self.boxes():
            if pid == permanent_id:
                return bit
        return None

    def mode_range(self, permanent_id):
        """(internal channel, low, high) of the first switch range for a mode, or None."""
        data = self.request(MODE_RANGES) or b""
        for i in range(0, len(data) - 3, 4):
            pid, aux_index, start, end = data[i:i + 4]
            if pid == permanent_id and end > start:
                return AUX1 + aux_index, 900 + start * 25, 900 + end * 25
        return None

    # --- telemetry --------------------------------------------------------
    def status(self):
        """(mode flags, arming disable flags, arming flag count) or None."""
        data = self.request(STATUS_EX, timeout=0.3)
        if not data or len(data) < 16:
            return None
        modes = struct.unpack_from("<I", data, 6)[0]
        extra = data[15]
        for i in range(extra):
            modes |= data[16 + i] << (32 + 8 * i)
        pos = 16 + extra
        if len(data) < pos + 5:
            return None
        return modes, struct.unpack_from("<I", data, pos + 1)[0], data[pos]

    def attitude(self):
        """(roll, pitch, yaw) in radians, or None."""
        data = self.request(ATTITUDE, timeout=0.2)
        if not data or len(data) < 6:
            return None
        roll, pitch, yaw = struct.unpack_from("<3h", data)
        return math.radians(roll / 10.0), math.radians(pitch / 10.0), math.radians(yaw)

    def altitude(self):
        """(altitude m, vertical speed m/s) from the baro+IMU estimate, or None."""
        data = self.request(ALTITUDE, timeout=0.2)
        if not data or len(data) < 6:
            return None
        altitude_cm, vario_cms = struct.unpack_from("<ih", data)
        return altitude_cm / 100.0, vario_cms / 100.0

    def imu(self):
        """(acc g [3], gyro deg/s [3]) or None."""
        data = self.request(RAW_IMU, timeout=0.2)
        if not data or len(data) < 18:
            return None
        v = struct.unpack_from("<9h", data)
        return [a / ACC_LSB_PER_G for a in v[0:3]], list(v[3:6])

    def battery(self):
        """(volts, amps, mAh drawn, rssi 0..1) or None."""
        data = self.request(ANALOG, timeout=0.2)
        if not data or len(data) < 9:
            return None
        _legacy_v, mah, rssi, amperage, voltage = struct.unpack_from("<BHHhH", data)
        return voltage / 100.0, amperage / 100.0, mah, rssi / 1023.0

    def rc(self):
        """Channels Betaflight is using, internal order (roll, pitch, yaw, throttle, aux...)."""
        data = self.request(RC, timeout=0.2) or b""
        if len(data) < 8:
            return []
        return list(struct.unpack_from("<%dH" % (len(data) // 2), data))


def arming_flag_names(flags, flag_count=None):
    names = []
    for i in range(32):
        if flags & (1 << i):
            if i < len(ARMING_FLAG_NAMES):
                names.append(ARMING_FLAG_NAMES[i])
            elif flag_count and i == flag_count - 1:
                names.append("ARMSWITCH")
            else:
                names.append("bit%d" % i)
    return names


class RcStream(threading.Thread):
    """Streams MSP_SET_RAW_RC at a fixed rate.

    While `paused`, nothing is sent at all. That is the hand-back mechanism: with MSP
    override active, Betaflight returns to the radio within ~100 ms of the data stopping.
    """

    def __init__(self, msp, rcmap, rate_hz=50.0, centre=1500, idle_throttle=1000):
        super().__init__(daemon=True)
        self.msp = msp
        self.rcmap = rcmap
        self.period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self.channels = [centre] * CHANNEL_COUNT
        self.channels[THROTTLE] = idle_throttle
        self.paused = True
        self._running = True
        self.sent = 0

    def set(self, channel, value):
        with self._lock:
            self.channels[channel] = int(round(value))

    def set_sticks(self, roll, pitch, yaw, throttle):
        with self._lock:
            self.channels[ROLL] = int(round(roll))
            self.channels[PITCH] = int(round(pitch))
            self.channels[YAW] = int(round(yaw))
            self.channels[THROTTLE] = int(round(throttle))

    def snapshot(self):
        with self._lock:
            return list(self.channels)

    def run(self):
        next_tick = time.monotonic()
        while self._running:
            if not self.paused:
                internal = self.snapshot()
                raw = [1500] * CHANNEL_COUNT
                for i, value in enumerate(internal):
                    raw[self.rcmap[i] if i < len(self.rcmap) else i] = value
                try:
                    self.msp.send(SET_RAW_RC, struct.pack("<%dH" % CHANNEL_COUNT, *raw))
                    self.sent += 1
                except serial.SerialException:
                    self.paused = True
            next_tick += self.period
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def stop(self):
        self.paused = True
        self._running = False
        self.join(timeout=1.0)
