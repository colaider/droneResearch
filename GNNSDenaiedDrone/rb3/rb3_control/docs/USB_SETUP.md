# USB on the RB3 Gen 2 (why the flight controller did not appear)

The two USB-A ports of the RB3 Gen 2 are behind a **Renesas uPD720201** PCIe xHCI controller.
That chip has no firmware EEPROM on this board, so Linux must upload a firmware blob at boot.
Ubuntu does not ship it (it is not redistributable), so a stock image shows:

```
xhci_hcd 0001:04:00.0: failed to load firmware renesas_usb_fw.mem: -2
xhci_hcd: probe of 0001:04:00.0 failed with error -2
```

With the controller unprobed, `lsusb` is empty, `/sys/bus/usb/devices/` is empty and no
`/dev/ttyACM*` ever appears — the USB-A ports are electrically dead. The SoC USB controllers
(`8c00000.usb`, `a600000.usb`) are in *device* mode for flashing/EUD, so they do not help.

## Fix

1. Download the **uPD720201/uPD720202 firmware** package from Renesas (login required):
   <https://www.renesas.com/us/en/products/interface/usb-switches-hubs/upd720201-usb-30-host-controller>
   The archive used here was `USB3-201-202-FW-20131112`, version 2.0.2.6, file `K2026090.mem`.
2. Copy it to the board **under the exact name the driver looks for**:

   ```bash
   scp K2026090.mem ubuntu@<board>:/tmp/renesas_usb_fw.mem
   sudo install -m 644 -o root -g root /tmp/renesas_usb_fw.mem /lib/firmware/renesas_usb_fw.mem
   ```

3. Re-probe without rebooting (or just reboot):

   ```bash
   echo 0001:04:00.0 | sudo tee /sys/bus/pci/drivers_probe
   ```

4. Serial port access for the normal user:

   ```bash
   sudo usermod -aG dialout $USER      # takes effect in a new session
   ```

## Verify

```bash
lsusb
# Bus 001 Device 003: ID 0483:5740 STMicroelectronics Virtual COM Port   <- the flight controller
ls -l /dev/ttyACM0
ros2 run rb3_control fc_check
```

Known-good result on this board: Betaflight 26.6.2, MSP API 1.48, board F405, plus the onboard
Genesys hubs and the ASIX AX88179 Ethernet adapter.

## Notes

* The firmware is read at probe time; once the file is in `/lib/firmware/` it is picked up on
  every boot automatically.
* Check which device node the FC got with `ls /dev/serial/by-id/` — with several USB serial
  devices attached, `ttyACM0` is not guaranteed to be the flight controller. Use a by-id path
  in `config/control.yaml` if that happens.
* Baud rate is irrelevant on a USB CDC port; it matters only for the `/dev/ttyHS2` UART fallback.
