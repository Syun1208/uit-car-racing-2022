import time

trafficSignsRegister = list()


class trafficSignsController:
    def __init__(self, mask, trafficSigns, bbox_area, confs):
        self.trafficSigns = trafficSigns
        self.bbox_area = bbox_area
        self.mask = mask
        self.angle = 0
        self.speed = 0
        self.confs = confs



    def __timer(self, time_set, angle):
        start = time.time()
        flag = False
        count = 0
        while not flag:
            self.angle = angle
            self.speed = 17
            count += 1
            if time.time() - start >= time_set or count > 15:
                flag = True

    def __delay_queue(self, time_delay):
        trafficSignsRegister.insert(0, self.trafficSigns)
        if len(trafficSignsRegister) > time_delay:
            trafficSignsRegister.pop(-1)

    def __call__(self, *args, **kwargs):
        self.__delay_queue(1000)
        self.speed = 10
        if self.bbox_area >= 3500 or self.confs >= 0.5:
            if self.trafficSigns == 'go straight' or 'go straight' in trafficSignsRegister:
                self.__timer(time_set=7, angle=0)
            elif self.trafficSigns == 'turn right' or self.trafficSigns == 'not left' or 'turn right' in trafficSignsRegister or 'not left' in trafficSignsRegister:
                self.__timer(time_set=7, angle=25)
            elif self.trafficSigns == 'turn left' or self.trafficSigns == 'not right' or 'turn left' in trafficSignsRegister or 'not right' in trafficSignsRegister:
                self.__timer(time_set=7, angle=-25)
        return self.angle, self.speed
