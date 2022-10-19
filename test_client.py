from unity_utils.unity_utils import Unity
import time
import logging
import argparse
import numpy as np
from tabulate import tabulate
from IPython.display import clear_output
from utils.controller import Controller, TrafficSignsController
from utils.traffic_signs_recognition import trafficSignsRecognition


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Input your port connection', default=11000)
    return parser.parse_args()


def main():
    args = parser_args()
    unity_api = Unity(args.port)
    unity_api.connect()
    frame = 0
    count = 0
    out_sign = "straight"
    flag_timer = 0
    fps = 20
    carFlag = 0
    trafficSigns = 'straight'
    pre_Signal = 'straight'
    signArray = np.zeros(15)
    noneArray = np.zeros(50)
    fpsArray = np.zeros(50)
    carArray = np.zeros(50)
    reset_seconds = 1.0
    try:
        while True:
            clear_output(wait=True)
            start_time = time.time()
            left_image, right_image = unity_api.get_images()
            image = np.concatenate((left_image, right_image), axis=1)
            '''--------------------------Controller--------------------------------'''
            if not flag_timer:
                frame += 1
                if frame % 1 == 0:
                    recognizeTrafficSigns = trafficSignsRecognition(image)
                    out_sign = recognizeTrafficSigns()
                if carFlag == 0:
                    if 50 <= frame < 100:
                        fpsArray[frame - 50] = fps
                    elif 100 <= frame < 120:
                        noneArray = np.zeros(int(np.mean(fpsArray) * reset_seconds))
                        carArray = noneArray[1:int(len(noneArray) / 2)]
                    elif frame > 150:
                        if out_sign == "none" or out_sign is None:
                            noneArray[1:] = noneArray[0:-1]
                            noneArray[0] = 0

                        else:
                            noneArray[1:] = noneArray[0:-1]
                            noneArray[0] = 1

                        if np.sum(noneArray) == 0:
                            out_sign = "straight"

                elif carFlag == 1:
                    if out_sign == "none" or out_sign is None or out_sign == "unknown":
                        carArray[1:] = carArray[0:-1]
                        carArray[0] = 0
                    else:
                        carArray[1:] = carArray[0:-1]
                        carArray[0] = 1
                    if np.sum(carArray) == 0:
                        out_sign = "straight"
                if out_sign != "unknown" and out_sign is not None and out_sign != "none":
                    if out_sign == "car_left" or out_sign == "car_right":
                        carFlag = 1
                    else:
                        carFlag = 0
                    trafficSigns = out_sign
                controller = Controller(image, trafficSigns)
                angle, speed = controller()
                if trafficSigns or trafficSigns != 'none' or trafficSigns != 'unknown':
                    trafficSignsController = TrafficSignsController(image, trafficSigns, speed)
                    trafficSigns, speed, angle = trafficSignsController()

                clear_output(wait=True)
                print("time: ", 1 / (time.time() - start_time))
                unity_api.show_images(left_image, right_image)
                data = unity_api.set_speed_angle(speed, angle)  # speed: [0:100], angle: [-25:25]
                clear_output(wait=True)
                text1 = ["SPEED", "ANGLE", 'TRAFFIC SIGNS']
                text2 = [["{:.2f}".format(data['Speed']), '{:.2f}'.format(data['Angle']), trafficSigns]]
                print(tabulate(text2, text1, tablefmt="pretty"))
    except Exception as error:
        text1 = ["NOTICE"]
        text2 = [["QUÁ GHÊ GỚM !"], ["VÀ ĐÂY LÀ PHOLOTINO !"]]
        logging.error(error)
        print(tabulate(text2, text1, tablefmt="pretty"))


if __name__ == "__main__":
    main()
