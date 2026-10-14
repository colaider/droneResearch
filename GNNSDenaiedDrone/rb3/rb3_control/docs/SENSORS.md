# RB3 Gen 2 on-board sensors (IMU, barometer, magnetometer)

The RB3 Gen 2 vision mezzanine carries an **ICM-42688-P** IMU, an **ICP-10111** barometer and
an **AKM** magnetometer. They are *not* Linux IIO devices and no device-tree change will make
them appear as such in the default configuration, because those buses do not belong to the
applications processor at all: they are wired to the **SSC (sensor core)** on the ADSP.

```
see_workhorse  ->  sscrpcd.service  ->  ADSP sensor PD (msm/adsp/sensor_pd)  ->  ICM-42688-P
  (userspace)        (fastrpc)              (Qualcomm SEE stack)                 ICP-10111
                                                                                 AKM mag
```

`rb3_control/see.py` wraps that: one `see_workhorse` process per sensor, its JSON event blocks
parsed into samples, published as ROS 2 topics by the `flight_controller` node.

> **Do not try to reach these sensors by enabling `spi12` / `i2c2` in the device tree.**
> GPIO48-51 are listed in `gpio-reserved-ranges` because they belong to the sensor core;
> removing that reservation makes `pinctrl-msm` touch TZ-protected pins and the board enters a
> **boot loop** that can only be fixed by reflashing `dtb_a`. The SEE path below needs no
> device-tree change whatsoever.

## Requirements on the board

All from the `ubuntu-qcom-iot/qcom-ppa` PPA, which the Ubuntu RB3 image already has configured:

```bash
sudo apt install qcom-sensors-test-apps     # see_workhorse, ssc_sensor_info
# dependencies, normally present already:
#   qcom-sensors-core qcom-sensors-services qcom-sensors-registry qcom-sensors-utils
systemctl is-active sscrpcd                 # must be active (it is enabled by default)
sudo -n true                                # see_workhorse needs root; passwordless sudo
```

`see_workhorse` run as a normal user exits successfully and produces **no samples at all**, so
check `sudo` first when a stream stays silent.

### DIP switches

Leave them at their **defaults**, which route the sensors to the sensor core:

| Switch | Default | Meaning |
| --- | --- | --- |
| mezz DIP1:1 `SENSOR_SPI_PASSTHRU` | ON | SSC SPI reaches the mezzanine IMU |
| mezz DIP1:2 `APPS_SPI_PASSTHRU` | OFF | apps-processor SPI not routed |
| mezz DIP2:3 `IMU_SPI_SELECT` | OFF | IMU on the SSC bus |
| mezz DIP2:6 `PRESSURE_COMPASS_SELECT` | OFF | baro/compass on the sensor-core I2C |
| mezz DIP1:4 / DIP1:5 | ON | pressure sensor / compass connected |

Setting `APPS_SPI_PASSTHRU`/`IMU_SPI_SELECT` routes the IMU to the applications processor,
which needs the device-tree work warned about above. There is no reason to do it.

## What the sensor core offers

```bash
sudo ssc_sensor_info | grep -E "^(NAME|TYPE|RATES)"
```

On this board, with the rates each sensor accepts (asking for an unsupported rate yields a
silent stream, so `see.py` snaps requests down to a supported value):

| Sensor | Driver | Units | Rates [Hz] |
| --- | --- | --- | --- |
| accel | `icm4x6xx` | m/s² | 12.5, 25, 50, 100, 200, **500** |
| gyro | `icm4x6xx` | rad/s | 12.5, 25, 50, 100, 200, **500** |
| pressure | `icp101xx` | hPa | 1, 5, 10, 15, 20, **25** |
| mag | `ak0991x` | µT | 1, 10, 20, 50, **100** |

QTI's fused algorithms (`rotv`, `game_rv`, `gravity`, `tilt`, `amd`, `pedometer`, ...) are
available over the same interface and can be added to `see.RATES` if ever needed.

## Topics

Published by the `flight_controller` node, independently of the MSP link, so they keep coming
while the radio has control.

| Topic | Type | Notes |
| --- | --- | --- |
| `imu` | `sensor_msgs/Imu` | gyro + accel; no orientation (`orientation_covariance[0] = -1`) |
| `mag` | `sensor_msgs/MagneticField` | tesla (SEE µT × 1e-6) |
| `pressure` | `sensor_msgs/FluidPressure` | pascal (SEE hPa × 100) |
| `baro_altitude` | `std_msgs/Float32` | metres above the start-up pressure level |

The `imu` topic is driven by the gyro stream with the newest accel sample merged in: the two
are separate processes and never share a timestamp.

`baro_altitude` is **relative**. With `baro_reference_hpa: 0.0` the node averages the first 20
pressure samples on the ground and reports height above that, which is what an altitude hold
wants. Set `baro_reference_hpa` to the local sea-level pressure instead for absolute altitude.

### Timestamps

Each sample carries the sensor core's own QTimer timestamp (19.2 MHz ticks), which `see.py`
converts to seconds and the node maps onto the ROS clock with a single offset measured at the
first sample. That matters: measured over 887 gyro samples at 200 Hz,

| Clock | mean Δt | std dev | min / max |
| --- | --- | --- | --- |
| SSC hardware timestamp | 5.025 ms | **0.003 ms** | 5.020 / 5.031 |
| time the reader thread got the line | 5.022 ms | 0.582 ms | 0.255 / 9.522 |

so using the host read time would hand any filter 200× more jitter than the sensor actually has.

## Checking it works

```bash
ros2 run rb3_control sensors                 # all four sensors, live, no ROS needed
ros2 run rb3_control fc_check                # includes an RB3 sensor section
ros2 topic hz /drone/imu                     # expect see_gyro_rate_hz
```

Expected at rest on a level surface: accel ≈ `[0, 0, 9.5]` m/s² (this board reads ~9.48, about
3 % low - calibrate if you need absolute values), gyro ≈ `[0.017, 0.005, -0.013]` rad/s bias,
pressure ≈ local QNH, mag ≈ 70 µT magnitude with a sizeable hard-iron offset.

Barometer noise measured on the bench: **±0.08 m**.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `see_workhorse not found` | `sudo apt install qcom-sensors-test-apps` |
| runs, exits `PASS`, zero samples | not root (`sudo -n true` fails), or an unsupported `-sample_rate` |
| `ssc_sensor_info` lists nothing | `sscrpcd` not running, or the ADSP did not boot - check `dmesg \| grep -i adsp` for `sensor_pd ... is up` |
| only some sensors listed | DIP switches routing them away from the SSC, or `PRESS_SENS_DISCONNECT` / `MAG_SENSOR_DISCONNECT` off |
| streams die and restart | ADSP subsystem restart; `see.SeeStream` respawns automatically and counts it in `restarts` |
