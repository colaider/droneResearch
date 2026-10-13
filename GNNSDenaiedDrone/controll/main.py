from GNNSDenaiedDrone.controll.communication.Rb3ReadSend import ReaderSender
import time


def main():
    reader = ReaderSender()                     
    try:
        while True:
            reader.step()                  
            reader.send(0, 0, 0, 10)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == "__main__":
    main()
