import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from utils.image_processing import imageProcessing


class trafficSignsRecognition:
    def __init__(self, mask):
        self.signs = ''
        self.mask_signs = mask
        self.colourCode = {'straight': [[0, 0, 254], [0, 0, 255]], 'no_straight': [[254, 0, 175], [255, 0, 180]],
                           'turn_left': [[0, 254, 0], [0, 254, 0]], 'no_left': [[0, 254, 254], [0, 255, 255]],
                           'turn_right': [[254, 0, 0], [255, 0, 0]], 'no_right': [[254, 0, 254], [255, 0, 255]]}

    def __call__(self, *args, **kwargs):
        # Straight
        straight_lower = np.array([0, 0, 254], np.uint8)
        straight_upper = np.array([0, 0, 255], np.uint8)
        straight_mask = cv2.inRange(self.mask_signs, straight_lower, straight_upper)

        contours, hierachy = cv2.findContours(straight_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'straight'
        # turn_right
        turn_right_lower = np.array([254, 0, 0], np.uint8)
        turn_right_upper = np.array([255, 0, 0], np.uint8)
        turn_right_mask = cv2.inRange(self.mask_signs, turn_right_lower, turn_right_upper)

        contours, hierachy = cv2.findContours(turn_right_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'turn_right'

        # turn_left
        turn_left_lower = np.array([0, 254, 0], np.uint8)
        turn_left_upper = np.array([0, 255, 0], np.uint8)
        turn_left_mask = cv2.inRange(self.mask_signs, turn_left_lower, turn_left_upper)

        contours, hierachy = cv2.findContours(turn_left_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'turn_left'

        # No_straight
        no_straight_lower = np.array([254, 0, 175], np.uint8)
        no_straight_upper = np.array([255, 0, 180], np.uint8)
        no_straight_mask = cv2.inRange(self.mask_signs, no_straight_lower, no_straight_upper)

        contours, hierachy = cv2.findContours(no_straight_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'no_straight'

        # no_right
        no_right_lower = np.array([254, 0, 254], np.uint8)
        no_right_upper = np.array([255, 0, 255], np.uint8)
        no_right_mask = cv2.inRange(self.mask_signs, no_right_lower, no_right_upper)

        contours, hierachy = cv2.findContours(no_right_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'no_right'

        # no_left
        no_left_lower = np.array([0, 254, 254], np.uint8)
        no_left_upper = np.array([0, 255, 255], np.uint8)
        no_left_mask = cv2.inRange(self.mask_signs, no_left_lower, no_left_upper)

        contours, hierachy = cv2.findContours(no_left_mask,
                                              cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                self.signs = 'no_left'

        return self.signs
