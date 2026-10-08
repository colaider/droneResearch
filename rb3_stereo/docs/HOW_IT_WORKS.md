# How the RB3 Gen 2 stereo camera was made to work

**Hardware:**
- Qualcomm RB3 Gen 2 Vision Kit (QCS6490).
- Two CMK OV9282 camera modules (1 MP, monochrome, global shutter) on the vision mezzanine
  connectors CAM0A and CAM0B.
- Ubuntu 24.04, kernel 6.8.0-1084-qcom, ROS 2 Jazzy.

**Goal:** a synchronized stereo pair at up to 120 fps.

**Final architecture:**
- **Open-source V4L2 camera path:** `qcom-camss` and the `ov9282` sensor driver, with three
  kernel-side changes:
  1. a 1-lane patch for the sensor driver;
  2. a "combo PHY" patch for camss, so both cameras share one CSI receiver;
  3. a small bring-up module for the second camera.
- **On top:** a Python capture layer with software frame sync, and a ROS 2 node.

---

## 1. Why not Qualcomm's camera stack

The board ships with Qualcomm's **CamX** stack. Its pieces are closed binaries:
- the `camera_qcm6490` kernel module;
- the `cam-server` daemon;
- the GStreamer source `qtiqmmfsrc`.

It runs the OV9282 at **30 fps maximum**. The GStreamer source advertises `framerate [0, 30]`,
there's no high-frame-rate path in this build, and the sensor configuration files are binary. So
the project moved to the upstream open-source path:
- `qcom-camss`: Qualcomm camera subsystem, V4L2 / media controller;
- `i2c-qcom-cci`: the camera I2C controller;
- the mainline `ov9282` sensor driver.

The Ubuntu kernel has all three, but the board's device tree doesn't describe the cameras for them.

## 2. Device tree for the open stack

1. **How to load a modified device tree.**
   - **GRUB:** its `devicetree` command crashed the Qualcomm UEFI with a "DXE panic".
   - **kexec:** booting a kernel with a new DT warm-reset the board during regulator setup.
   - **What works:** the firmware loads a *multi-DTB* file from the `dtb_a` partition, so that
     file is patched directly.
     - `drivers/devicetree/dtb_flash.sh` backs the partition up to `dtb_a_backup.img`, applies the overlay to the
       matching DTB entries, and writes it back. `dtb_restore.sh` undoes it.
2. **The overlay (`drivers/devicetree/ov9282-cam0a.dtso`)** enables `camss` for CAM0A:
   - **Clocks:** both the old and the new clock names, because Ubuntu's camss backport asks for
     some of each.
   - **Power domains:** the correct `power-domain-names`; the shipped DT misspells the property.
   - **PHY supplies:** the CSI PHY power supplies.
   - **CCI clock:** 37.5 MHz.
   - **Sensor node:** 24 MHz MCLK0, reset on GPIO20, and the CSI endpoint.
3. **The dead IMX577 camera** was removed from the DT. camss waits for *every* camera in its DT
   graph, so the media device never appeared while it was there.
4. **CamX was blocked:**
   - `module_blacklist=camera_qcm6490` on the kernel command line;
   - `install /bin/false` in modprobe.d.

   Both stacks drive the same hardware and clashed on the I2C controller interrupt.

## 3. Sensor driver: the module has one MIPI lane

**Symptom:** the sensor reported "streaming" but no frame arrived. The CSI receiver (CSID) only
showed a lane-0 FIFO overflow.

**Finding the cause:** CamX could still stream, so it served as a reference.
1. With CamX's kernel debug logging on, its CSI PHY register dump showed lane mask `0x81`: clock
   lane plus **one** data lane. The CSID was set to 1 active lane. The CMK module is wired for one
   lane; the mainline driver only supports two.
2. CamX's I2C traffic to the sensor was decoded from its CCI command words: 111 register writes.
   Diffing those against the mainline driver gave the 1-lane settings:
   - `0x030e = 0x06`: PLL2 system divider;
   - `0x4837 = 0x15`: MIPI clock period;
   - `0x3662`: 8- or 10-bit mode for one lane.

**The patch** (`drivers/ov9282`):
- module parameter `lanes=1`;
- writes those three registers after the mode table;
- reports the correct pixel rate for one lane (80 / 96 MHz), so blanking and fps calculations come
  out right.

The camss endpoint uses `data-lanes = <0>`.

**Result:** 640x400 @ 120 fps; 1280x800 @ 60 fps (10-bit) or 70 fps (8-bit). One lane at
~800 Mbit/s is the bandwidth limit at full resolution.

## 4. The second camera (CAM0B)

**Wiring:** read from Qualcomm's own DT node for that slot (`qcom,cam-sensor7`):

| Signal | Connection |
|---|---|
| I2C | second camera I2C controller `cci1`, master 0 (disabled in the DT) |
| Clock | MCLK4 on GPIO68 |
| Reset | GPIO149 |
| Supply | the same 1.8 V as CAM0A |
| Data | CSI0 PHY *lane 2*, with *lane 3 as its clock* ("2+1 combo mode"; CAM0A uses lane 0 + the clock lane) |

Vision mezzanine switch **DIP2-1 must be OFF**. When ON, all four CSI0 lanes go to CAM0A.

1. **`cam0b_bridge.ko`**: brings up CAM0B without changing the DT or rebooting.
   - Creates the platform device for the disabled `cci1` node, so `i2c-qcom-cci` binds and provides
     an I2C bus (`/dev/i2c-20`).
   - Powers the sensor with the regulator, clock, reset GPIO and pin mux listed in Qualcomm's node.
2. **camss combo-mode patch** (`drivers/camss`, parameters `combo_phy=0 combo_alias=1`). Upstream
   camss allows one sensor per PHY and one receiver per PHY output.
   - **Second capture path:** the unused entity `msm_csiphy1` becomes an *alias* that drives
     CSIPHY0's registers and clocks.
   - **Shared PHY power:** power-up and reset happen only for the first user; the stream count
     covers both sensors.
   - **Lane enable:** the register holds the union of both sensors' lanes, so it reads `0x81` for
     A alone and `0xD1` for both. Stopping one camera only removes its own lanes.
   - **Combo register table:** the PHY gets Qualcomm's combo table, which configures lane 3 as a
     clock lane. It's taken from the vendor driver source and matches CamX's register dump.
   - **Receiver:** CSID1 reads PHY 0, lane 2.
3. **Sensor B has no V4L2 driver.** Its registers are *cloned* over I2C from the running CAM0A
   sensor, so both run the identical mode.
4. **Gotcha:** CAM0B still held CamX's values in registers that a soft reset doesn't restore.
   - The PLL multiplier `0x030d` made it run at exactly 5/6 speed (100 instead of 120 fps).
   - MIPI control `0x4800` produced CRC and ECC errors and broken frames.

   The clone list now includes these registers. Result: both cameras at 119.95 fps, error-free.

## 5. Synchronization

1. **Hardware frame sync (FSIN on connector pin 27).** The board connects CAM0A-27 to CAM0B-27.
   The SoC can't see this pin; it isn't on the LS2 header.
   - **The test:** sensor B as master (FSIN output) and sensor A as slave. A locked to B's frame
     rate exactly, so the modules do route FSIN.
   - **How the lock works:** the slave's own frame time must be slightly *longer* than the
     master's. Otherwise it fits two frames per pulse.
   - **Why it isn't used:** switching sync on while streaming cuts a frame short. Upstream camss
     can't recover from that: the image engine's write master stops completing buffers and the
     capture blocks forever. Starts failed 4 times out of 6.
2. **Software genlock (used).**
   - Both sensors' clocks come from the same SoC reference, so their frame periods are identical
     and the offset only needs setting once. Free-running drift measured ~2 ppm.
   - `align()` makes CAM0B's frames a few lines longer or shorter (VTS) for a counted number of
     frames until both frames end together. A background thread re-checks every 5 s.
   - Frames are never cut short, so no capture path stalls.
   - **Result:** median offset ≤ ±40 µs, p5–p95 mostly ≤ ±80 µs.
   - The smallest correction step is coarser than one line, about 80 µs at 120 fps, so that's the
     practical resolution.
3. **Timestamp fixes.**
   - Frame times come from the kernel's frame-done interrupt (CLOCK_MONOTONIC).
   - Both cameras' interrupts were on CPU0. That serialized them and skewed camera B's timestamps
     by ~85 µs. `rb3-stereo-setup` moves them to the little A55 cores 2 and 3, and `run_stereo.sh`
     pins the node to cores 0–3, so the big A78 cores 4–7 stay free for the vision code.
   - The capture tool's log was being read byte by byte, which lagged 0.5 s. It's now read in
     chunks: ~7 ms.

## 6. ROS 2 node

- **Capture:** `v4l2-ctl` per camera; OpenCV can't open camss' multi-planar devices.
- **Pairing:** frames are paired by kernel timestamp, and both images get the same stamp: the
  exposure centre on the ROS clock.
- **Settings:** all are ROS parameters and can be changed live. A resolution change restarts the
  pipelines.
- **Auto-exposure:** in software; there's no ISP on this raw path.
- **Watchdog:** restarts the pipelines if no pair arrives for 2 s.
- **Drop-in for the simulation:** `StereoSubscriber` reproduces `get_two_frames()`, `res`, `fov`
  and `camera_saperation` from the topics. `StereoPublisher` lets the Genesis simulation publish
  the same topics.
- **Large images:** Fast DDS's default 512 KB shared-memory segments dropped most 1 MB images
  (1280x800 arrived at 20 of 60 fps). `config/fastdds.xml` raises the segment size.

---

## Hardware and platform issues

| # | Issue | Symptom | Fix |
|---|---|---|---|
| 1 | Qualcomm CamX caps the OV9282 at 30 fps | GStreamer refuses > 30 fps | open-source camss stack |
| 2 | UEFI crashes on GRUB `devicetree`; kexec resets the board | board doesn't boot the new DT | patch the multi-DTB in partition `dtb_a` |
| 3 | DT clock names / misspelled `power-domain-names` | camss probe fails (`vfe0 -2`) | overlay with both clock-name sets and correct names |
| 4 | DT still lists the absent IMX577 | no `/dev/media` (camss waits forever) | remove its endpoint and node |
| 5 | CamX and camss loaded together | interrupt clash on the camera I2C controller | blacklist CamX |
| 6 | CMK module wired with 1 MIPI lane; mainline driver needs 2 | sensor streams, 0 frames, CSID FIFO overflow | patched `ov9282` (`lanes=1`) and `data-lanes = <0>` |
| 7 | 1 lane ≈ 800 Mbit/s | 1280x800 maxes out at ~60–70 fps | use 640x400 for 120 fps |
| 8 | CAM0B's I2C controller (`cci1`) is disabled in the DT | sensor unreachable | `cam0b_bridge.ko` |
| 9 | CAM0A and CAM0B share CSI0 (2+1 combo); camss can't | second camera can't stream | camss combo patch |
| 10 | Vision mezzanine DIP2-1 ON | CAM0B gets no data lanes | DIP2-1 OFF |
| 11 | Leftover CamX registers in sensor B survive soft reset | 100 fps; ECC/CRC errors, no frames | clone PLL / MIPI registers too |
| 12 | FSIN pin 27 not reachable from the SoC | SoC can't trigger or see sync | sensor-to-sensor test |
| 13 | camss has no recovery from a malformed frame | capture blocks after FSIN start/stop | software genlock; node watchdog |
| 14 | Both camera interrupts on CPU0 | ~85 µs timestamp skew between the cameras | IRQ affinity in `rb3-stereo-setup` |
| 15 | Fast DDS 512 KB shared-memory segments | 1280x800 subscribers get 1/3 of the frames | `config/fastdds.xml` |

## Potential issues with this setup

**Software and maintenance**
- **Kernel updates break the camera.** The three drivers are out-of-tree and unsigned, and built
  for one kernel. After `apt upgrade` installs a new `linux-qcom`, rerun `drivers/install_board.sh`, or
  hold the kernel: `sudo apt-mark hold linux-image-qcom linux-headers-qcom linux-qcom`.
- **Firmware flashing undoes the DT patch.** The device tree in `dtb_a` is patched; a QDL or BSP
  flash restores the stock DT. Keep `dtb_a_backup.img` and `dtb_flash.sh`.
- **Qualcomm's camera stack is disabled.** No `qtiqmmfsrc` and no IMX577. Reverting needs
  `dtb_restore.sh` and a reboot.
- **The kernel side is a hack, not upstream.**
  - CAM0B is created at runtime, not from a DT node.
  - The `msm_csiphy1` entity is repurposed, so a camera on the CAM1 port can't be used with combo
    mode on.
  - A proper solution is a DT node for CAM0B plus combo support in camss upstream.
- **No error recovery in camss.** A corrupted frame (loose connector, EMI, ESD) can stall a capture
  path. The node detects this and restarts both pipelines, losing ~2–3 s of video. Watch the log
  for "restarting the camera pipelines".
- **Sync is software-maintained.**
  - It's typically ±40 µs, with transients of a few hundred µs right after start or a resolution
    change, for up to ~5 s.
  - Each correction makes a few CAM0B frames ~15 µs longer or shorter. Strict-timing consumers can
    check `/stereo/status` → `sync_offset_us` and `pair_offset_us`.
- **Exposure changes can land one frame apart** between the two sensors. Auto-exposure is simple
  and runs at ~10 Hz.
- **Timestamps are interrupt times.** The exposure centre is computed from frame end − readout −
  exposure/2. It's accurate to roughly ±50 µs.
- **Python and v4l2-ctl pipes.** About 70–85 % of one core at full rate (of 8), and ~10 ms of
  latency before ROS transport. A C++ node using V4L2 directly would cut both.
- **ROS 2 launch and signals.** Stop with SIGINT / Ctrl-C. On SIGTERM, launch exits and leaves the
  node running. `respawn` restarts a crashed node.

**Simulation vs. real camera**
- **Optics.** `camera_info` uses the simulation's values (85° vertical FOV, 5 cm baseline) until
  the cameras are calibrated. The real lens has a different FOV and distortion. The vision code
  must use `camera_info` (`res`, `fov`, `camera_saperation` from `StereoSubscriber`), not
  hard-coded values.
- **Image format.** The simulation renders 500x600 colour; the real images are 640x400 mono with
  noise, lens distortion, motion blur and limited dynamic range. `StereoSubscriber` expands mono to
  BGR so the code path is the same.
- **Mounting.** The simulation's cameras look down, since it uses altitude as feature depth. Mount
  the real pair the same way, and rigidly: the baseline must not flex.

**Hardware**
- **Bandwidth:** one lane per camera, so 1280x800 can't go above ~60–70 fps.
- **CAM0B signal margin:** its path runs through the mezzanine's DIP-switch lane mux. It was
  error-free at ~800 Mbit/s in the tests, but margins under vibration and temperature are unknown.
  Corruption would show up as pipeline restarts.
- **Connectors under vibration:** the board-to-board connectors and flex cables need strain relief
  and locking.
- **Pin 27:** it's tied between the two modules. Never configure both sensors as FSIN outputs: that
  would make them drive the same line against each other. The current setup leaves both as inputs.
- **Thermal:** the RB3 runs two camera pipelines, the node and the vision code. Check temperature
  in an enclosed airframe.
- **Network:** raw images are ~61 MB/s at 640x400 @ 120 fps. Never subscribe to raw topics over
  Wi-Fi; use `preview/compressed`.
  - `ROS_DOMAIN_ID` must match (42).
  - DDS discovery doesn't cross subnets or default WSL2 networking.
- **Access:** temporary passwordless sudo (`/etc/sudoers.d/90-claude-temp`) and an SSH key were
  added for remote work. Remove them when you no longer need remote work done on the board.
