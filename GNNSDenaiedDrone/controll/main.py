import argparse

from controllers.control_loop import ControlLoop


def main():
    parser = argparse.ArgumentParser(description="Run GNSS-denied drone control.")
    parser.add_argument(
        "--show-camera",
        action="store_true",
        help="display the live left/right stereo camera stream in a window",
    )
    args = parser.parse_args()
    loop = ControlLoop(show_camera=args.show_camera)
    loop.run()


if __name__ == "__main__":
    main()
