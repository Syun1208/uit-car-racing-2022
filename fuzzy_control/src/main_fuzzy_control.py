from fuzzy import Fuzzy
import time
import numpy as np
driver = Fuzzy()
center_array1 = [80, 80,80,80,80,80,80,80,80,80,80,80,80]
center_array2 = [320, 50, 60, 55, 70, 70, 80, 80, 96, 100, 110, 120, 130, 140, 150, 160, 180, 200, 190, 188, 185, 170, 165, 160, 160, 160]
center_array3 = [320, 50, 60, 55, 70, 80, 200]
def fuzzy_controller():
	for center in center_array2:
		t = time.time()
		print("center", center)
		error_center = center - 160
		steer = -6*driver.run_fuzzy_controller(error_center)
		print('steer fps', steer, int(1/(time.time()-t)))
if __name__ == '__main__':
	fuzzy_controller()

