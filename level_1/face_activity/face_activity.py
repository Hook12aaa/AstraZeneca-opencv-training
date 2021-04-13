import cv2.cv2 as cv2
import numpy as np


class face_activity():
    mask = cv2.imread("./face_activity/dog.png")
    cascade = cv2.CascadeClassifier("./face_activity/haarcascade_frontalface_default.xml")
    
    def __init__(self,force_camera = True, Camera_nunber = 3) -> None:
        self.__activiate_camera(Camera_nunber,force_camera)


    def __overlay_mask(self,face: np.array, mask: np.array) -> np.array:
        """Add the mask to the provided face, and return the face with mask.

        Args:
            face (np.array): The image around the face
            mask (np.array): The overlay onto of the face

        Returns:
            np.array: The changed image with mask overlayed on face

        """
        mask_h, mask_w, _ = mask.shape
        face_h, face_w, _ = face.shape

        # Resize the mask to fit on face
        factor = min(face_h / mask_h, face_w / mask_w)
        new_mask_w = int(factor * mask_w)
        new_mask_h = int(factor * mask_h)
        new_mask_shape = (new_mask_w, new_mask_h)
        resized_mask = cv2.resize(mask, new_mask_shape)

        # Add mask to face - ensure mask is centered
        face_with_mask = face.copy()
        non_white_pixels = (resized_mask < 250).all(axis=2)
        off_h = int((face_h - new_mask_h) / 2)
        off_w = int((face_w - new_mask_w) / 2)
        face_with_mask[off_h: off_h+new_mask_h, off_w: off_w+new_mask_w][non_white_pixels] = \
            resized_mask[non_white_pixels]

        return face_with_mask




    def __activiate_camera(self,num:int,user_force_camera:bool):
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
    
        if user_force_camera:
            self.cap = cv2.VideoCapture(num)

    def convert_to_grey(self,frame = None)-> np.array:
        """convert video image into grey

        Args:
            frame (OpenCV Image): Not needed to declare but can if you want

        Returns:
            np.array: returns black and white image
        """
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.blackwhite = cv2.equalizeHist(gray)
        return self.blackwhite


    def get_video_frame(self) ->np.array:
        """parse frame for camera into a return. Gets a frame

        Returns:
            np.array: your image from the camera
        """
        __ , self.frame = self.cap.read()
        self.frame_h, self.frame_w, _ = self.frame.shape
        return self.frame

    def crop_image(self,x:int,y:int,w:int,h:int):
        """crop the image to around the person

        Args:
            x (int): X  cordinates of the image
            y (int): Y cordinates of the image
            w (int): The size of the image vertically
            h (int): The size of the image horizontally
        """
        self.y0, self.y1 = int(y - 0.25*h), int(y + 0.75*h)
        self.x0, self.x1 = x, x + w
    
    def is_out_of_frame(self) -> bool:
        """Will return true if you are outside of the camera view

        Returns:
            bool: True if outside of image
        """
        if self.x0 < 0 or self.y0 < 0 or self.x1 > self.frame_w or self.y1 > self.frame_h:
            return True
        else:
            return False

    def detect_human(self,noir_imagr:np.array):
        """Detect Human will return locations of people

        Args:
            noir_imagr (OpenCV_image): the black and white image

        Returns:
            rect: all postions of each person in frame
        """
        cv2.imshow("noir_imagr",noir_imagr)

        rects = self.cascade.detectMultiScale(noir_imagr, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

        return rects

    def show_image(self,frame= None):
        """will display image as a spearte window when called

        Args:
            frame (OpenCV_image): Can be empty if needed, just show passing the frame into another area
        """
        cv2.imshow('frame', frame)


    def apply_mask(self,frame = None):
        """apply the mask to your image

        Args:
            frame (opecv_image): The frame from the camera
        """
        self.frame[self.y0:self.y1, self.x0: self.x1] = self.__overlay_mask(self.frame[self.y0: self.y1, self.x0: self.x1], self.mask)


    def end_program(self):
        """decalare at the end of the program"""
        self.cap.release()
        cv2.destroyAllWindows()


