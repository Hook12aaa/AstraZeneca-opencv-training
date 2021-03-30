import cv2.cv2 as cv2

import cv2.cv2 as cv2
import numpy as np

cap = cv2.VideoCapture("./footage/drift_shakey_pratice.mp4")


class drift_remove():
    def __init__(self):
        self.previous_frame = 0
        self.frame_count = 0
        self.buffer_of_x = []

    def __remove_drift(self):
        """This gets a dictation of the drone by reading the buffer of past recordings """
        total = sum(self.buffer_of_x) / 10
        self.buffer_of_x = []
        if total > 0:
            return "LEFT", total
        if total < 0:
            return "RIGHT", total * -1
        if total == 0:
            return "STRAIGHT", total

    def __catch_x(self, grey_img):
        """Takes (x,_) of a sharp point and adds it to an array"""
        corners = cv2.goodFeaturesToTrack(grey_img, 1, 0.01, 10)
        x, y = np.int0(corners).ravel()
        total = (x - self.previous_frame)
        if total < 15 and total > -15 and total != 1:
            self.buffer_of_x.append(total)
        self.previous_frame = x
        self.frame_count += 1

    def stabilize_drone(self, grey_img):
        """frame needed for processing. returns every 20 iteration"""
        if self.frame_count == 20:
            self.frame_count = 0
            return self.__remove_drift()
        else:
            self.__catch_x(grey_img)
            return False

