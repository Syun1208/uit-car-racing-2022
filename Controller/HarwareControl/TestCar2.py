import UITCar
import time
Car = UITCar.UITCar()

Car.setMotorMode(0)
while 1:
    Speed = float(input("Toc Do: "))
    Car.setSpeed_cm(Speed)
    Angle = float(input("Goc: "))
    Car.setAngle(Angle)