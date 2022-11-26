import UITCar
import time
Car = UITCar.UITCar()

def BTN1_Func(chann):
    print("Pressed 1")
    
def BTN2_Func(chann):
    print("Pressed 2")
    
def BTN3_Func(chann):
    print("Pressed 3")

Car.setSpeed_cm(50)

CurrentPos = Car.getPosition_cm()
while(Car.getPosition_cm() - CurrentPos > 1000):
    time.sleep(0.1)
    print("Car Pos ", Car.getPosition_cm())

print("Code Done")