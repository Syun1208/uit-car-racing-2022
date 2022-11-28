import UITCar
import time
Car = UITCar.UITCar()

def BTN1_Func(chann):
    print("Pressed 1")
    
def BTN2_Func(chann):
    print("Pressed 2")
    
def BTN3_Func(chann):
    print("Pressed 3")

Car.regBTN(1,BTN1_Func)
Car.setMotorMode(0)
Car.setSpeed_cm(90)

CurrentPos = Car.getPosition_cm()
Car.setAngle(30)
print("Start Pos ", CurrentPos)
while(Car.getPosition_cm() - CurrentPos < 1000):
    time.sleep(0.1)
    print("Car Pos ", Car.getPosition_cm())
Car.setSpeed_cm(0)
print("End Pos ", Car.getPosition_cm())

print("Motor Mode 0 Done")

print("Start Motor Mode 1")

Car.setMotorMode(1)
Car.setAngle(-30)
Car.setPosition_cm(1000, 70)
# time.sleep(0.5)
while(Car.getSpeed_cm()!= 0):
    time.sleep(0.1)
    print("Car Speed ", Car.getSpeed_cm())
    print("Car Pos ", Car.getPosition_cm())

print("Motor Mode 1 Done")