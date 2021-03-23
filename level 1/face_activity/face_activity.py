import cv2.cv2 as cv2
import numpy as np


class face_activity():
    mask = cv2.imread(".face_activity/dog.png")
    cascade = cv2.CascadeClassifier(".face_activity/haarcascade_frontalface_default.xml")
    
    def __init__(self,force_camera = True, Camera_nunber = 3) -> None:
        self.__activiate_camera(Camera_nunber,force_camera)


    def __overlay_mask(face: np.array, mask: np.array) -> np.array:
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
        if user_force_camera:
            for i in num:
                self.cap = cv2.VideoCapture(i)
                if self.cap is not None and self.cap.isOpened():
                    break
                else:
                    continue
    
        if not user_force_camera:
            self.cap = cv2.VideoCapture(num)

    def video_turn_grey(self)-> np.array:
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blackwhite = cv2.equalizeHist(gray)
        return blackwhite


    def get_video_camera(self) ->np.array:
        __ , self.frame = self.cap.read()
        self.frame_h, self.frame_w, _ = self.frame.shape
        return self.frame

    def crop_image(self,x:int,y:int,w:int,h:int):
        self.y0, self.y1 = int(y - 0.25*h), int(y + 0.75*h)
        self.x0, self.x1 = x, x + w
    
    def is_out_of_frame(self) -> bool:
        if self.x0 < 0 or self.y0 < 0 or self.x1 > self.frame_w or self.y1 > self.frame_h:
            return True
        else:
            return False


    def show_image(self):
        cv2.imshow('frame', self.frame)


    def apply_mask(self):
        self.frame[self.y0:self.y1, self.x0: self.x1] = self.__overlay_mask(self.frame[self.y0: self.y1, self.x0: self.x1], self.mask)


    def end_program(self):
        self.cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    camera = face_activity()