from image_processing import imageProcessing
import time
from sklearn.ensemble import RandomForestRegressor
import numpy as np

t = time.time()
error_arr = np.zeros(5)
list_angle = np.zeros(5)
trafficSignsRegister = list()


class ControllerByTrafficSigns(imageProcessing):
    def __init__(self, mask, trafficSigns):
        super(ControllerByTrafficSigns, self).__init__(mask, trafficSigns)
        self.mask = mask
        self.trafficSigns = trafficSigns
        self.minLane = 0
        self.maxLane = 0
        self.error = 0

    def computeMinMaxLane(self, scale=40):
        arr_normal = []
        height = self.mask.shape[0] - scale
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y.any() == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 // 3, self.mask.shape[1] * 2 // 3]
        self.minLane = min(arr_normal)
        self.maxLane = max(arr_normal)
        # center = int((minLane + maxLane) / 2)
        # error = int(self.mask.shape[1] / 2) - center
        return self.minLane, self.maxLane

    def straight(self):
        center = int((self.minLane + self.maxLane) / 2)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def turnLeft(self):
        center = int((self.minLane + self.maxLane) * 1 / 3)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def turnRight(self):
        center = int((self.minLane + self.maxLane) * 2 / 3)
        error = int(self.mask.shape[1] / 2) - center
        return error

    def noTurnLeft(self):
        center = 0
        error = int(self.mask.shape[1] / 2) - center
        return error

    def noTurnRight(self):
        center = 0
        error = int(self.mask.shape[1] / 2) - center
        return error

    def __call__(self, *args, **kwargs):
        _, __, __, horizontal_check_left = self.houghLines(self.left_mask)
        _, __, __, horizontal_check_right = self.houghLines(self.right_mask)
        self.minLane, self.maxLane = self.computeMinMaxLane()
        trafficSignsRegister.insert(0, self.trafficSigns)
        if self.trafficSigns == 'trai' or 'trai' in trafficSignsRegister:
            self.error = self.turnLeft()
        elif self.trafficSigns == 'phai' or 'phai' in trafficSignsRegister:
            self.error = self.turnRight()
        elif self.trafficSigns == 'camthang':
            if horizontal_check_right:
                self.error = self.turnRight()
                trafficSignsRegister.insert(0, 'phai')
            elif horizontal_check_left:
                self.error = self.turnLeft()
                trafficSignsRegister.insert(0, 'trai')
        elif self.trafficSigns == 'camphai' or 'camphai' in trafficSignsRegister:
            self.error = self.noTurnRight()
        elif self.trafficSigns == 'camtrai' or 'camtrai' in trafficSignsRegister:
            self.error = self.noTurnLeft()
        # elif self.trafficSigns == 'thang' or 'thang' in trafficSignsRegister:
        #     self.error = self.straight()
        else:
            self.error = self.straight()
        if len(trafficSignsRegister) > 90:
            trafficSignsRegister.pop(-1)
        return self.error


class Controller(ControllerByTrafficSigns):
    def __init__(self, image, trafficSigns, current_speed):
        if not trafficSigns:
            trafficSigns = [-1]
        self.trafficSigns = list(['camtrai', 'camphai', 'camthang', 'trai', 'phai', 'thang', 'none'])[
            int(trafficSigns[0])]
        ControllerByTrafficSigns.__init__(self, image, self.trafficSigns)
        self.error = ControllerByTrafficSigns.__call__(self)
        self.current_speed = current_speed
        self.mask = image
        self.scale = 1
        self.__CHECKPOINT = 45
        self.__LANE_WIDTH = 55

    def __reduceSpeed(self, speed):
        if self.current_speed > 20:
            return -2
        else:
            return speed

    def findingLane(self, scale=40):
        arr_normal = []
        height = self.mask.shape[0] - scale
        lineRow = self.mask[height, :]
        for x, y in enumerate(lineRow):
            if y.any() == 255:
                arr_normal.append(x)
        if not arr_normal:
            arr_normal = [self.mask.shape[1] * 1 // 3, self.mask.shape[1] * 2 // 3]
        minLane = min(arr_normal)
        maxLane = max(arr_normal)
        center = int((minLane + maxLane) / 2)
        width = maxLane - minLane
        # Consider center's condition
        if width < self.__LANE_WIDTH:
            if width < int(self.mask[1] / 2):
                center -= self.__LANE_WIDTH - width
            else:
                center += self.__LANE_WIDTH - width
        error = int(self.mask.shape[1] / 2) - center
        return error

    @staticmethod
    def __PID(error, scale=1, p=0.43, i=0, d=0.05):
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
        # angle = self.__optimizeFuzzy(angle)
        if abs(angle) > 50:
            angle = np.sign(angle) * 50
        return - int(angle) * scale

    @staticmethod
    def __conditionalSpeed(error):
        list_angle[1:] = list_angle[0:-1]
        list_angle[0] = abs(error)
        list_angle_train = np.array(list_angle).reshape((-1, 1))
        predSpeed = np.dot(list_angle, - 0.1) + 20
        # reg = LinearRegression().fit(list_angle_train, speed)
        reg = RandomForestRegressor(n_estimators=40, random_state=1).fit(list_angle_train, predSpeed)
        predSpeed = reg.predict(np.array(list_angle_train))
        return predSpeed[0]

    def __call__(self, *args, **kwargs):
        error = self.findingLane()
        print('Traffic Sign: ', self.trafficSigns)
        if not self.trafficSigns or self.trafficSigns != 'none' or self.trafficSigns != 'thang':
            error = self.error
            self.scale = 2
        angle = self.__PID(error, self.scale)
        speed = self.__conditionalSpeed(error)
        speed = self.__reduceSpeed(speed)
        # angle = angle * 60 / 25
        return angle, speed, self.trafficSigns
