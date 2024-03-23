import queue
import time
import cv2
import argparse
import threading
import keyboard
import logging
from queue import Queue

logging.basicConfig(filename="C:/Users/Asus TUF/PycharmProjects/pythonProject5/log/lab4.log", level=logging.DEBUG)

q = Queue()

def sensor_worker(sensor):
    while True:
        global exit_flag
        value = sensor.get()
        q.put((sensor._name, value))
        if exit_flag:
            break

class Sensor:
    def get(self):
        raise NotImplementedError('Subclasses must implement method get()')

class SensorX(Sensor):
    def __init__(self, delay, name):
        self._name = name
        self._delay = delay
        self._data = 0

    def get(self):
        time.sleep(self._delay)
        self._data += 1
        return self._data

class SensorCam(Sensor):
    def __init__(self, name, width, height):
        self._name = name
        self._width = width
        self._height = height

        self._cam = cv2.VideoCapture(name)
        if self._cam.isOpened() is False:
            logging.error(f'camera {name} could not be used')
            global exit_flag
            exit_flag = True
        else:
            self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get(self):
        _, frame = self._cam.read()

        return frame

    def __del__(self):
        self._cam.release()

class WindowImage:
    def __init__(self, rate):
        self._rate = rate

    def show(self, dict):
        if dict[name_cam] is not None:
            cv2.imshow("Video", dict[name_cam])
            if (dict['s0'] is not None) or (dict['s1'] is not None) or (dict['s2'] is not None):
                cv2.putText(dict[name_cam], f"Sensor 0: {dict['s0']}",(20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(dict[name_cam], f"Sensor 1: {dict['s1']}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(dict[name_cam], f"Sensor 2: {dict['s2']}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                logging.error('data could not be received from one of the sensors')
            cv2.waitKey(self._rate)
        else:
            logging.error('The image could not be displayed')

    def __del__(self):
        cv2.destroyAllWindows()

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('name', type = int, help = 'name of camera in system')
    parser.add_argument('width', type = int, help = 'width of window')
    parser.add_argument('height', type = int, help = 'heigth of window')
    parser.add_argument('rate', type = int, help = 'frame rate')

    args = parser.parse_args()

    return args

args = arg_parser()
name_cam = args.name
cam = SensorCam(name_cam, args.width, args.height)
window = WindowImage(args.rate)

# cam = SensorCam(0, 640, 640)
# window = WindowImage(1)

sensor0 = SensorX(0.01, 's0')
sensor1 = SensorX(0.1, 's1')
sensor2 = SensorX(1, 's2')

thread0 = threading.Thread(target=sensor_worker, args=(cam,))
thread1 = threading.Thread(target=sensor_worker, args=(sensor0,))
thread2 = threading.Thread(target=sensor_worker, args=(sensor1,))
thread3 = threading.Thread(target=sensor_worker, args=(sensor2,))

thread0.start()
thread1.start()
thread2.start()
thread3.start()

dict = {'s0':None, 's1':None, "s2":None, name_cam:None}

exit_flag = False

def exit_program(event=None):
    global exit_flag
    exit_flag = True

keyboard.on_press_key("q", exit_program)

while not exit_flag:
    a, b = q.get()
    dict[a] = b
    window.show(dict)

thread0.join()
thread1.join()
thread2.join()
thread3.join()




