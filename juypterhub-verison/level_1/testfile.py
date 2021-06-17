#
import cv2.cv2 as cv2
import numpy as np
import sys
from ipywebrtc import CameraStream,ImageRecorder
from ipywidgets import Output
from PIL import Image
import io
import numpy as np
from matplotlib import pyplot as plt




class look_up():
    out = Output()

    
    @out.capture()
    def on_value_changed(_):
        global camera_state
        im = Image.open(io.BytesIO(image_recorder.image.value)) 
        camera_state = np.array(im)

    def run_program(self):
        global image_recorder
        camera = CameraStream.facing_user(audio=False)
        image_recorder = ImageRecorder(stream=camera)
        image_recorder.image.observe(look_up.on_value_changed, 'value')
        image_recorder.recording = True
        return camera_state





class face_activity():
    
    mask = cv2.imread("./face_activity/dog.png")
    cascade = cv2.CascadeClassifier("./face_activity/haarcascade_frontalface_default.xml")

    def __init__(self) -> None:
        
        print("Camera Object is activated")


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
        attempt = look_up()
        frame =  attempt.run_program()
        convert = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        self.frame = cv2.cvtColor(convert, cv2.COLOR_RGBA2RGB)
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

    def detect_people(self,noir_imagr:np.array):
        """Detect Human will return locations of people

        Args:
            noir_imagr (OpenCV_image): the black and white image

        Returns:
            rect: all postions of each person in frame
        """
 
        rects = self.cascade.detectMultiScale(noir_imagr, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
        return rects
    

    def show_image(self,frame= None):
        """will display image as a spearte window when called

        Args:
            frame (OpenCV_image): Can be empty if needed, just show passing the frame into another area
        """     
        plt.imshow(self.frame)
        plt.show()

    def change_to_brown_dog_face(self):
        """Autosetted to brown dog face but will show brown dog again"""
        self.mask = cv2.imread("./face_activity/dog.png")
    
    def change_to_dalmation_dog_face(self) -> None:
        """replaces your overal with a dalmation instead"""
        self.mask = cv2.imread("./face_activity/dog_2.png")

    def change_to_black(self) -> None:
        """replaces the overlay with a black image
        """
        self.mask = cv2.imread("./face_activity/black.png")
    def apply_mask(self,frame = None) -> None:
        """apply the mask to your image

        Args:
            frame (opecv_image): The frame from the camera
        """
        self.frame[self.y0:self.y1, self.x0: self.x1] = self.__overlay_mask(self.frame[self.y0: self.y1, self.x0: self.x1], self.mask)

    def end_program(self):
        """decalare at the end of the program"""
        print("end_program() was called: Closing Application")
        self.cap.release()
        cv2.destroyAllWindows()

