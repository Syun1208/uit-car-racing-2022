from unity_utils.unity_utils import Unity
import time
import numpy as np
from tabulate import tabulate
from IPython.display import clear_output
from utils.controller import Controller, TrafficSigns

unity_api = Unity(11000)
unity_api.connect()


def main():
    try:
        while True:
            clear_output(wait=True)
            start_time = time.time()
            left_image, right_image = unity_api.get_images()
            image = np.concatenate((left_image, right_image), axis=1)
            '''--------------------------Controller--------------------------------'''
            controller = Controller(image)
            angle, speed = controller()
            print("time: ", 1 / (time.time() - start_time))
            # unity_api.show_images(left_image, right_image)
            data = unity_api.set_speed_angle(speed, angle)  # speed: [0:100], angle: [-25:25]
            text1 = ["SPEED", "ANGLE"]
            text2 = [["{:.2f}".format(data['Speed']), '{:.2f}'.format(data['Angle'])]]
            print(tabulate(text2, text1, tablefmt="pretty"))
    except Exception as error:
        text1 = ["NOTICE"]
        text2 = [["QUÁ GHÊ GỚM !"], ["VÀ ĐÂY LÀ PHOLOTINO !"]]
        print(error)
        print(tabulate(text2, text1, tablefmt="pretty"))


if __name__ == "__main__":
    main()
