from communication.reader import Reader
import rclpy

rclpy.init()
node = rclpy.create_node("reader")     # <-- the node you pass to Reader
reader = Reader(node)


def main():
    print(reader.get_gyro)

if __name__ == "__main__":
    while 1:
        main()