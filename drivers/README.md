# drivers – kernel side of the RB3 Gen 2 stereo camera

Two OV9282 global-shutter cameras on the vision mezzanine (CAM0A + CAM0B), driven by upstream
camss instead of Qualcomm's CamX. The ROS 2 node that uses them is in `../rb3_stereo`. How and why
it works: `../rb3_stereo/docs/HOW_IT_WORKS.md`.

| Folder / file | What it is |
|---|---|
| `ov9282/` | Ubuntu's ov9282 driver with a 1-MIPI-lane mode (`lanes=1`); the camera modules run one lane |
| `camss/` | Ubuntu 6.8 qcom-camss with a CSIPHY0 2+1 combo mode, so CAM0A and CAM0B share the PHY (`combo_phy=0 combo_alias=1`) |
| `cam0b_bridge/` | Creates the disabled `cci1` I2C bus and powers CAM0B at runtime (no device-tree node needed) |
| `install_board.sh` | Builds the three modules for the running kernel and installs them, `rb3-stereo-setup` and its service. `--enable-boot` loads them at every boot |
| `rb3-stereo-setup` | Loads the modules and moves the camera interrupts to CPU 2/3. Run by `rb3-stereo-setup.service` at boot |
| `devicetree/` | `dtb_flash.sh` patches the firmware device tree (`dtb_a` partition) so camss owns CAM0A and blocks CamX; `dtb_restore.sh` undoes it |

## Setup order on a freshly flashed board

1. `drivers/devicetree/dtb_flash.sh` (asks before writing), then reboot. It keeps the original
   partition as `dtb_a_backup.img` next to the script. Keep that file: it is the way back to the
   stock device tree (`dtb_restore.sh`). It is not in git (64 MB).
2. `scripts/setup_rb3.sh`. This runs `install_board.sh --enable-boot` and sets up the ROS workspace.
3. Vision mezzanine switch DIP2-1 OFF.

After a kernel update, rerun `scripts/setup_rb3.sh` (or `sudo drivers/install_board.sh --enable-boot`).

Check that the drivers loaded: `systemctl status rb3-stereo-setup` should show
`camss combo mode on, CAM0B on i2c-<n>`. Turn off loading at boot:
`sudo systemctl disable rb3-stereo-setup`.
