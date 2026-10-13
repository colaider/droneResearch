from communication.reader import Reader
import time


def main():
    reader = Reader()                      # creates and owns its node
    try:
        while True:
            reader.step()                  # all the plumbing: spin + latch
            print("gyro:", reader.get_gyro(),
                  "accel:", reader.get_accel(),
                  "baro:", reader.get_baro())
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
