from deploy.image_processing import trafficSignsController
import time
import numpy as np

t = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)


class Controller(trafficSignsController):
    def __init__(self, mask, maxSpeed, trafficSigns, Car, area, confs):
        self.trafficSigns = list(['go straight','turn left','turn right','not left', 'not right', 'none'])[int(trafficSigns)]
        trafficSignsController.__init__(self, mask, self.trafficSigns, area, confs)
        self.mask = mask
        self.Car = Car
        self.current_speed = self.Car.getSpeed_rad()
        self.maxSpeed = maxSpeed

    def __reduceSpeed(self, speed):
        if self.current_speed > 27.1:
            return 15
        else:
            return speed

    def findingLane(self, scale=24):
        arr_normal = []
        height = int(self.mask.shape[0] - scale)
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 // 3, self.mask.shape[1] * 2 // 3]
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        width = maxLane - minLane
        # if width < 55:
        #     if center < int(self.mask.shape[1] / 2):
        #         center -= 55 - width
        #     else:
        #         center += 55 - width
        error = int(self.mask.shape[1] / 2) - center
        return error

    @staticmethod
    def __PID(error, scale=1, p=1.2, i=0, d=0):
        global t
        global error_arr
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error * p
        delta_t = time.time() - t
        t = time.time()
        D = (error - error_arr[1]) / delta_t * d
        I = np.sum(error_arr) * delta_t * i
        angle = P + I + D
        if abs(angle) > 44:
            angle = np.sign(angle) * 44
        return int(angle) * scale

    def __speedByError(self, error):
        predSpeed = abs(error) * - 0.09 + self.maxSpeed
        return predSpeed

    def __speedByAngle(self, angle):
        predSpeed = int(abs(angle) * - 0.09 + self.maxSpeed)
        return predSpeed

    def __call__(self, *args, **kwargs):
        error = self.findingLane()
        # print('Traffic Sign: ', self.trafficSigns)
        self.Car.OLED_Print('Traffic Sign: {}'.format(self.trafficSigns), 0)
        angle = self.__PID(error)
        speed = self.__speedByError(error)
        if self.trafficSigns != 'none':
            print(self.trafficSigns)
            angle, speed = trafficSignsController.__call__(self)
        if abs(angle) > 25:
            speed = 13.6
        # speed = self.__reduceSpeed(speed)
        return angle, speed
