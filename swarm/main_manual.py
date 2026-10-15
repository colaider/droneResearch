import os
import cv2
import numpy as np
import genesis as gs
from swarm.scene_builders.outdoorScene import OutdoorScene
from swarm.droneClasses.droneManualCTRL import DroneManualCTRL
from swarm.utilities.rtPlotter import LivePlotter

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
DRONE_URDF = os.path.join(REPO_ROOT, "genesis_drones/robots/assets/drone_urdf/custom_racer.urdf")

dt = 0.01  # Simulation timestep

# Camera window names/positions -- matches DroneStruct.camera_show(), which imshows
# frames under str(i) for i in range(len(postpocessed_frames)) (2 stereo frames -> "0", "1").
CAM_WINDOWS = [("0", 40, 100), ("1", 620, 100)]


def main():
    gs.init(backend=gs.cpu)

    # Pre-create the camera windows at a known, visible screen position. cv2's
    # default window placement/size can end up off-screen or behind other windows
    # depending on the window manager; pinning them up front avoids that.
    for name, x, y in CAM_WINDOWS:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 500, 400)
        cv2.moveWindow(name, x, y)

    print("\n📍 Building outdoor obstacle-avoidance scene...")
    scene = OutdoorScene(
        dt=dt,
        gravity=(0, 0, -9.81),
        show_viewer=True,
        camera_pos=(3, 3, 2),
        camera_lookat=(0, 0, 0.5),
    )
    scene.populate_outdoor_obstacles(
        num_trees=14,
        num_buildings=5,
        area=(24.0, 24.0),
        clear_radius=3.0,
        seed=1,
    )

    drone = scene.add_drone(
        name="drone_1",
        urdf_path=DRONE_URDF,
        position=(0, 0, 0),
        orientation=(0, 0, 0),
    )

    controller = DroneManualCTRL(drone_entity=drone, dt=dt, start_z=1.2)

    # Static camera on purpose: viewer.follow_entity() recomputes the camera position
    # and look-at point every frame to chase the drone, which reads as constant
    # panning/zooming of the world even when you aren't doing anything.

    scene.build()

    # Genesis's own viewer binds w/s/a/d/r/f to camera/debug shortcuts (toggle world-frame
    # axes, save screenshot, auto-rotate camera, wireframe, video recording, face normals).
    # If a keypress lands on the main viewer window instead of a camera window, those fire
    # instead of drone control -- strip them so stray focus there is a no-op rather than
    # confusing (e.g. 'w' toggling the world-frame axes on/off).
    for kb_name in ("world_frame", "save_image", "camera_rotation", "wireframe", "record_video", "face_normals"):
        try:
            scene.scene.viewer.remove_keybind(kb_name)
        except Exception as exc:
            print(f"[warn] could not remove viewer keybind '{kb_name}': {exc}")

    plotter = LivePlotter(
        plots={
            'Accelerometer (m/s²)': ['ax', 'ay', 'az'],
            'Gyroscope (rad/s)':    ['gx', 'gy', 'gz'],
            'Actual pos (m)':       ['x', 'y', 'z'],
        },
        buffer_size=1000,
        title='IMU Live',
    )

    print("\n🎮 Manual control -- click into one of the camera windows (titled '0' or '1'), then use:")
    print("   w / s : move forward / backward")
    print("   a / d : strafe left / right")
    print("   r / f : ascend / descend")
    print("   q / e : yaw left / right")
    print("   ESC   : quit\n")

    printed_first_frame = False
    last_report_step = -100

    for step in range(100_000_000):
        controller.step(step)

        # Show the two stereo CV windows (and read keyboard input from them) every
        # step, not just once manual control kicks in, so they're visible immediately.
        key = controller.drone.camera_show()
        if not printed_first_frame and controller.drone.postpocessed_frames:
            print(f"[camera] first frame(s) shown: {len(controller.drone.postpocessed_frames)} window(s)")
            printed_first_frame = True
        if key != -1:
            # Diagnostic: prints on ANY keypress cv2 receives, regardless of what it maps
            # to. If pressing a key never prints this line, the key event is not reaching
            # the app at all (a window-focus issue), not a mapping problem in handle_key().
            print(f"[key] raw={key}  char={chr(key & 0xFF)!r}")
        if key == 27:  # ESC
            break

        reading = controller.drone.imu.read()
        acc = reading.lin_acc.numpy()   # array shape (3,)
        gyro = reading.ang_vel.numpy()  # array shape (3,)
        pos = controller.get_position()
        plotter.push('Accelerometer (m/s²)', acc)
        plotter.push('Gyroscope (rad/s)', gyro)
        plotter.push('Actual pos (m)', np.atleast_1d(pos))
        plotter.update()

        if controller.start(step) == 1:
            controller.handle_key(key)
            controller.fly()
            if step - last_report_step >= 100:  # ~once per second at dt=0.01
                print(f"[step {step}] pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})  setpoint={controller.setpoint.round(2)}  acc={acc.round(2)}")
                last_report_step = step

        scene.step()

    print("\n✅ Simulation ended.")
    scene.close()


if __name__ == "__main__":
    main()
