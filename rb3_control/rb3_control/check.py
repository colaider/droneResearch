"""Read-only preflight check of the RB3 <-> flight controller USB link.

Verifies the things the control node needs, and prints the exact Betaflight CLI commands
for anything missing. Sends no stick commands and never arms.

    python3 -m rb3_control.check [--port /dev/ttyACM0] [--watch]
    ros2 run rb3_control fc_check -- --watch
"""
from __future__ import annotations

import argparse
import sys
import time

from . import msp as m
from . import see

USB_VCP = 20            # Betaflight serial port identifier of the USB virtual COM port


def port_name(identifier):
    """Betaflight serialPortIdentifier_e -> label.

    Older firmware numbered UARTs 0..7; 2026.6 (API 1.48) reports them as 51..58.
    """
    if identifier < 8:
        return "UART%d" % (identifier + 1)
    if identifier == USB_VCP:
        return "USB"
    if 30 <= identifier <= 32:
        return "SOFTSERIAL%d" % (identifier - 29)
    if 40 <= identifier <= 48:
        return "LPUART%d" % (identifier - 39)
    if 50 <= identifier <= 58:
        return "UART%d" % (identifier - 50)
    return "port%d" % identifier


def check(link, verbose=True):
    results, fixes = [], []

    identity = link.identity()
    if identity is None:
        print("No MSP answer. Is the FC on this port, and is nothing else using it?")
        return False
    print("Firmware : %(variant)s %(version)s   MSP API %(api)s   board %(board)s" % identity)

    samples = []
    for _ in range(20):
        start = time.monotonic()
        if link.request(m.API_VERSION, timeout=0.5) is not None:
            samples.append((time.monotonic() - start) * 1000.0)
    if samples:
        print("Link     : %d/20 replies, round trip avg %.1f ms, max %.1f ms"
              % (len(samples), sum(samples) / len(samples), max(samples)))
        results.append((len(samples) == 20, "link stable (20/20 replies)"))

    ports = link.serial_ports()
    if ports:
        print("Ports    : " + "  ".join("%s=%d" % (port_name(i), f) for i, f in sorted(ports.items())))
    usb_msp = bool(ports.get(USB_VCP, m.FUNCTION_MSP) & m.FUNCTION_MSP)
    results.append((usb_msp, "MSP available on the USB port (always on in Betaflight)"))

    rcmap = link.rc_map()
    order = "".join("AERT12345678"[rcmap.index(i)] for i in range(len(rcmap))) if rcmap else "?"
    print("Channels : map %s (order used in MSP_SET_RAW_RC)" % order)

    boxes = {pid: (bit, name) for pid, bit, name in link.boxes()}
    arm = link.mode_range(m.BOX_ARM)
    override = link.mode_range(m.BOX_MSP_OVERRIDE)
    angle = link.mode_range(m.BOX_ANGLE)

    results.append((arm is not None,
                    "ARM on AUX%d (%d-%d)" % (arm[0] - 3, arm[1], arm[2]) if arm
                    else "ARM assigned to a radio switch"))
    results.append((m.BOX_MSP_OVERRIDE in boxes,
                    "firmware has MSP OVERRIDE (USE_RX_MSP_OVERRIDE)"))
    results.append((override is not None,
                    "MSP OVERRIDE on AUX%d (%d-%d)" % (override[0] - 3, override[1], override[2])
                    if override else "MSP OVERRIDE assigned to a radio switch"))
    if angle:
        print("Angle    : ANGLE mode on AUX%d (%d-%d)" % (angle[0] - 3, angle[1], angle[2]))
    if m.BOX_MSP_OVERRIDE in boxes and override is None:
        fixes.append("aux 5 50 3 1700 2100 0 0        # MSP OVERRIDE on AUX4 high")
        fixes.append("set msp_override_channels_mask = 15   # roll+pitch+yaw+throttle")

    status = link.status()
    if status:
        names = m.arming_flag_names(status[1], status[2])
        print("Arming   : " + (" ".join(names) if names else "no blocking flags"))
        armed = bool(status[0] & (link.box_bit(m.BOX_ARM) or 0))
        print("State    : %s" % ("ARMED" if armed else "disarmed"))

    attitude, altitude, battery = link.attitude(), link.altitude(), link.battery()
    if attitude:
        print("Attitude : roll %+.1f deg  pitch %+.1f deg  yaw %.0f deg"
              % (attitude[0] * 57.2958, attitude[1] * 57.2958, attitude[2] * 57.2958))
    if altitude:
        print("Altitude : %.2f m   vario %+.2f m/s" % altitude)
    if battery:
        print("Battery  : %.2f V   %.2f A   %d mAh" % battery[:3])

    results += check_sensors()

    print()
    for ok, text in results:
        print("  [%s] %s" % ("OK" if ok else "!!", text))
    if fixes:
        print("\nRun these in the Betaflight CLI, then `save`:")
        for line in fixes:
            print("   " + line)
    print()
    return all(ok for ok, _ in results)


def check_sensors(timeout=4.0):
    """Are the RB3's own sensors reachable through the sensor core? (see see.py)"""
    print()
    if not see.available():
        print("RB3 sensors: see_workhorse not installed")
        return [(False, "RB3 sensor core: sudo apt install qcom-sensors-test-apps")]

    sensors = see.SensorSet({"accel": 200.0, "gyro": 200.0, "pressure": 25.0, "mag": 50.0})
    sensors.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not all(s.alive() for s in sensors.streams.values()):
        time.sleep(0.2)

    results = []
    for name, stream in sorted(sensors.streams.items()):
        sample = stream.peek()
        if sample is None:
            print("RB3 %-8s: no samples" % name)
            results.append((False, "RB3 %s: no samples - is sscrpcd running and sudo passwordless?"
                            % name))
            continue
        values = sample[2]
        unit = {"accel": "m/s^2", "gyro": "rad/s", "pressure": "hPa", "mag": "uT"}[name]
        print("RB3 %-8s: %s %s  @ %g Hz"
              % (name, " ".join("%+9.3f" % v for v in values[:3]), unit, stream.rate))
        results.append((True, "RB3 %s streaming at %g Hz" % (name, stream.rate)))
    sensors.stop()
    return results


def watch(link):
    """Live view: switch positions, attitude, altitude, arming flags. Ctrl+C to stop."""
    arm = link.mode_range(m.BOX_ARM)
    override = link.mode_range(m.BOX_MSP_OVERRIDE)
    arm_bit = link.box_bit(m.BOX_ARM) or 0
    print("\nLive view - Ctrl+C to stop\n")
    while True:
        status = link.status()
        rc = link.rc()
        attitude = link.attitude()
        altitude = link.altitude()
        if not status or not rc:
            print("\rno telemetry", end="", flush=True)
            time.sleep(0.2)
            continue
        armed = bool(status[0] & arm_bit)
        switches = []
        for label, mode in (("ARM", arm), ("OVERRIDE", override)):
            if mode and len(rc) > mode[0]:
                on = mode[1] <= rc[mode[0]] < mode[2]
                switches.append("%s AUX%d=%d %s" % (label, mode[0] - 3, rc[mode[0]],
                                                    "ON" if on else "off"))
        flags = " ".join(m.arming_flag_names(status[1], status[2])) or "none"
        print("\r%-8s | %s | roll %+6.1f pitch %+6.1f | alt %5.2f m | flags: %-28s"
              % ("ARMED" if armed else "disarmed", "  ".join(switches),
                 (attitude[0] * 57.2958) if attitude else 0.0,
                 (attitude[1] * 57.2958) if attitude else 0.0,
                 altitude[0] if altitude else 0.0, flags),
              end="", flush=True)
        time.sleep(0.1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--watch", action="store_true", help="live view after the checks")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        link = m.Msp(args.port, args.baud)
    except Exception as exc:                      # noqa: BLE001 - report any open failure plainly
        raise SystemExit("cannot open %s: %s\nIs the user in the dialout group?" % (args.port, exc))
    time.sleep(0.3)
    try:
        ok = check(link)
        if args.watch:
            watch(link)
    except KeyboardInterrupt:
        print()
    finally:
        link.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
