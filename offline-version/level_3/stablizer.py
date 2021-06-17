import cv2.cv2 as cv2
import numpy as np


class stabilize():
    def __init__(self,force_camera = False,Camera_nunber = 0):
        self.previous_frame = 0
        self.frame_count = 0
        self.buffer_of_x = []
        self.__activiate_camera(force_camera,Camera_nunber)



    def __activiate_camera(self,user_force_camera:bool,num:int):
        """private function that forces camera to speed up loading time

        Args:
            num (int): Camera's Device numb eg 1 or 2....
            user_force_camera (bool): True at default will open camera on auto
        """
        if not user_force_camera:
            for i in range(num):
                self.cap = cv2.VideoCapture(i)
                if self.cap is not None and self.cap.isOpened():
                    break
                else:
                    continue
    
        if  user_force_camera:
            self.cap = cv2.VideoCapture(num)

    def get_video_frame(self) ->np.array:
        """parse frame for camera into a return. Gets a frame

        Returns:
            np.array: your image from the camera
        """
        __ , self.frame = self.cap.read()
        return self.frame

    def __get_total(self):
        """Sums up movement in a range of frames of the video to find the avarage

        Returns:
            float: the total possible shake to happen
        """
        total = round(sum(self.buffer_of_x) / 10,1)
        self.buffer_of_x = []
        return total 

    def __catch_x(self, grey_img):
        """ Catch the x cord from the frame of the video. Will use goodFeaturesToTrack

        Args:
            grey_img (np.array): The Grey Scale that you would like to track
        """
        corners = cv2.goodFeaturesToTrack(grey_img, 1, 0.01, 10)
        x, y = np.int0(corners).ravel()
        total = (x - self.previous_frame)
        if total < 15 and total > -15 and total != 1:
            self.buffer_of_x.append(total)
        self.previous_frame = x
        self.frame_count += 1

    def __get_metrics(self, grey_img):
        """frame needed for processing. returns every 20 iteration

        Args:
            grey_img (np.array) [description]

        Returns:
            float: The shake amount dectected in the image
        """
        if self.frame_count == 10:
            self.frame_count = 0
            return self.__get_total()
        else:
            self.__catch_x(grey_img)



    def show_image(self,frame= None) -> None:
        """will display image as a spearte window when called

        Args:
            frame (OpenCV_image): Can be empty if needed, just show passing the frame into another area
        """
        cv2.imshow('frame', self.frame)


    def get_shake(self,grey:np.array) ->None:
        """get shake value and will print it out

        Args:
            grey (np.array): the grey of the video
        """
        r = self.__get_metrics(grey)
        if r != None:
            print(r)

