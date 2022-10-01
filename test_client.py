from unity_utils.unity_utils import Unity
import cv2

import time
import numpy as np
from utils.controller import Controller

unity_api = Unity(11000)
unity_api.connect()


def main():
    while True:
        start_time = time.time()
        left_image, right_image = unity_api.get_images()
        image = np.concatenate((left_image, right_image), axis=1)
        '''--------------------------Controller--------------------------------'''
        controller = Controller(image)
        angle, speed = controller()
        print("time: ", 1 / (time.time() - start_time))
        unity_api.show_images(left_image, right_image)
        data = unity_api.set_speed_angle(speed, angle)  # speed: [0:100], angle: [-25:25]
        print(data)


if __name__ == "__main__":
    main()
