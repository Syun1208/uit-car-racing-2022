import socket
from unity_utils.unity_utils import Unity
import time
import logging
import matplotlib.pyplot as plt
import signal
import pandas as pd
import cv2
import requests
import playsound
import eyed3
from pygame import mixer
import argparse
import numpy as np
import os
import sys
from pathlib import Path
from tabulate import tabulate
from multiprocessing import Process
from IPython.display import clear_output
from utils.controller import Controller

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, help='Input your port connection', default=11000)
    parser.add_argument('--url', type=str, help='url downloading music',
                        default='https://stream.nixcdn.com/NhacCuaTui2029/DoanTuyetNangDiFrexsRemix-PhatHuyT4-7022889.mp3?st=SQkSrqlEECJcdathICmHdg&e=1666795517')
    return parser.parse_args()


def playMusicPHOLOTINO(url, name_music):
    # args = parser_args()
    # url = args.url
    # video = pafy.new(url)
    # best = video.getbest()
    # play_url = best.url
    # Instance = vlc.Instance()
    # player = Instance.media_player_new()
    # Media = Instance.media_new(play_url)
    # try:
    #     Media.get_mrl()
    #     player.set_media(Media)
    #     player.pause()
    #     player.play()
    # except Exception as bug:
    #     logging.error(bug)
    #     player.stop()
    try:
        # url = args.url
        downloaded_file_location = ROOT / name_music
        r = requests.get(url)
        with open(downloaded_file_location, 'wb') as f:
            f.write(r.content)
        mixer.init()
        mixer.music.load(ROOT / name_music)
        mixer.music.set_volume(1)
        mixer.music.play()
    except Exception as bug:
        logging.error(bug)
        # playsound.playsound(name_music, True)
        mixer.init()
        mixer.music.load(ROOT / name_music)
        mixer.music.set_volume(1)
        mixer.music.play()


def main():
    args = parser_args()
    unity_api = Unity(args.port)
    unity_api.connect()
    print('BẤM ĐI BTC !')
    # frame = 0
    # out_sign = "straight"
    # flag_timer = 0
    # fps = 20
    # speed = 20
    # carFlag = 0
    # trafficSigns = 'straight'
    # noneArray = np.zeros(50)
    # fpsArray = np.zeros(50)
    # carArray = np.zeros(50)
    # reset_seconds = 1.0

    while True:
        clear_output(wait=True)
        start_time = time.time()
        left_image, right_image = unity_api.get_images()
        image = np.concatenate((left_image, right_image), axis=1)

        '''--------------------------Controller--------------------------------'''
        # if not flag_timer:
        #     frame += 1
        #     if frame % 1 == 0:
        #         recognizeTrafficSigns = trafficSignsRecognition(image)
        #         out_sign = recognizeTrafficSigns()
        #     if carFlag == 0:
        #         if 50 <= frame < 100:
        #             fpsArray[frame - 50] = fps
        #         elif 100 <= frame < 120:
        #             noneArray = np.zeros(int(np.mean(fpsArray) * reset_seconds))
        #             carArray = noneArray[1:int(len(noneArray) / 2)]
        #         elif frame > 150:
        #             if out_sign == "none" or out_sign is None:
        #                 noneArray[1:] = noneArray[0:-1]
        #                 noneArray[0] = 0
        #
        #             else:
        #                 noneArray[1:] = noneArray[0:-1]
        #                 noneArray[0] = 1
        #
        #             if np.sum(noneArray) == 0:
        #                 out_sign = "straight"
        #
        #     elif carFlag == 1:
        #         if out_sign == "none" or out_sign is None or out_sign == "unknown":
        #             carArray[1:] = carArray[0:-1]
        #             carArray[0] = 0
        #         else:
        #             carArray[1:] = carArray[0:-1]
        #             carArray[0] = 1
        #         if np.sum(carArray) == 0:
        #             out_sign = "straight"
        #     if out_sign != "unknown" and out_sign is not None and out_sign != "none":
        #         if out_sign == "car_left" or out_sign == "car_right":
        #             carFlag = 1
        #         else:
        #             carFlag = 0
        #         trafficSigns = out_sign
        # if trafficSigns != 'none' or trafficSigns != 'unknown' or trafficSigns is not None:
        # recognizeTrafficSigns = trafficSignsRecognition(image)
        # trafficSigns = recognizeTrafficSigns()
        controller = Controller(image)
        angle, speed = controller()
        # else:
        #     trafficSignsController = TrafficSignsController(image, trafficSigns, speed)
        #     trafficSigns, speed, angle = trafficSignsController()
        # print("time: ", 1 / (time.time() - start_time))
        # unity_api.show_images(left_image, right_image)
        data = unity_api.set_speed_angle(speed, angle)  # speed: [0:100], angle: [-25:25]
        # print('Angle: ', data['Angle'])
        # print('Traffic Signs: ', trafficSigns)
        # text1 = ["SPEED", "ANGLE", 'TRAFFIC SIGNS']
        # text2 = [["{:.2f}".format(data['Speed']), '{:.2f}'.format(data['Angle']), trafficSigns]]
        # print(tabulate(text2, text1, tablefmt="pretty"))


if __name__ == "__main__":
    try:
        args = parser_args()
        p1 = Process(target=playMusicPHOLOTINO(args.url, 'pholotino.mp3'))
        p1.start()
        # p2 = Process(target=getLyricsMusic())
        # p2.start()
        p3 = Process(target=main())
        p3.start()
        p1.join()
        # p2.join()
        p3.join()
        p1.terminate()
        # p2.terminate()
        p3.terminate()
    except Exception as e:
        logging.error(e)
        text1 = ["NOTICE"]
        text2 = [["QUÁ GHÊ GỚM !"], ["VÀ ĐÂY LÀ PHOLOTINO !"]]
        print(tabulate(text2, text1, tablefmt="pretty"))
    # df = pd.read_csv('data/data_linux2.csv')
    # df = df.to_dict('list')
    # plt.scatter(df['error'], df['expected_speed'], label='Expected Speed', color='red')
    # plt.plot(df['error'], df['predicted_speed'], label='Predicted Speed', color='blue')
    # plt.title('Random Forest Regression Model Map Linux 2')
    # plt.xlabel('Error')
    # plt.ylabel('Speed')
    # plt.legend(loc='best')
    # plt.savefig('map_linux2.png')
