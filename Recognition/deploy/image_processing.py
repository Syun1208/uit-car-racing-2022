class trafficSignsController:
    def __init__(self, mask, trafficSigns, bbox_area):
        self.trafficSigns = trafficSigns
        self.bbox_area = bbox_area
        self.mask = mask
        self.scale = 1
        self.angle = 0
        self.error = 0
        self.horizontal_check_left = False
        self.horizontal_check_right = False

    def computeError(self, scale, minLane, maxLane):
        center = int((minLane + maxLane) * scale)
        width = maxLane - minLane
        if width < 55:
            if center < int(self.mask.shape[1] / 2):
                center -= 55 - width
            else:
                center += 55 - width
        return int(self.mask.shape[1] / 2) - center

    def __straight(self, minLane, maxLane):
        if 10 <= minLane <= 80:
            return self.computeError(7 / 8, minLane, maxLane)
        elif 80 < maxLane <= 160:
            return self.computeError(1 / 8, minLane, maxLane)
        else:
            return self.computeError(0, minLane, maxLane)

    def __turnLeft(self, minLane, maxLane):
        return self.computeError(1 / 3, minLane, maxLane)

    def __turnRight(self, minLane, maxLane):
        return self.computeError(2 / 3, minLane, maxLane)

    def __call__(self, minLane, maxLane, *args, **kwargs):
        trafficSignsRegister.insert(0, self.trafficSigns)
        if self.bbox_area > 3700:
            if self.trafficSigns == 'thang' or 'thang' in trafficSignsRegister:
                self.error = self.__straight(minLane, maxLane)
                self.angle = 0
            elif self.trafficSigns == 'trai' or 'trai' in trafficSignsRegister:
                self.error = self.__turnLeft(minLane, maxLane)
                self.scale = 2
                self.angle = - 40
            elif self.trafficSigns == 'phai' or 'phai' in trafficSignsRegister:
                self.error = self.__turnRight(minLane, maxLane)
                self.scale = 2
                self.angle = 40
            elif self.trafficSigns == 'camtrai' or 'camtrai' in trafficSignsRegister:
                if 80 < maxLane <= 160:
                    trafficSignsRegister.insert(0, 'phai')
                    self.error = self.__turnRight(minLane, maxLane)
                    self.angle = 40
                else:
                    trafficSignsRegister.insert(0, 'thang')
                    self.error = self.__straight(minLane, maxLane)
                self.scale = 2
            elif self.trafficSigns == 'camphai' or 'camphai' in trafficSignsRegister:
                if 10 <= minLane <= 80:
                    trafficSignsRegister.insert(0, 'trai')
                    self.error = self.__turnLeft(minLane, maxLane)
                    self.angle = - 40
                else:
                    trafficSignsRegister.insert(0, 'thang')
                    self.error = self.__straight(minLane, maxLane)
                self.scale = 2
            elif self.trafficSigns == 'camthang' or 'camthang' in trafficSignsRegister:
                if 10 <= minLane <= 80 or self.horizontal_check_left:
                    trafficSignsRegister.insert(0, 'trai')
                    self.error = self.__turnLeft(minLane, maxLane)
                elif 80 < maxLane <= 160 or self.horizontal_check_right:
                    trafficSignsRegister.insert(0, 'phai')
                    self.error = self.__turnRight(minLane, maxLane)
                self.scale = 2
            else:
                self.error = self.__straight(minLane, maxLane)
            if len(trafficSignsRegister) > 1000:
                trafficSignsRegister.pop(-1)
        return self.error, self.scale