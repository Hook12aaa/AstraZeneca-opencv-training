
import cv2.cv2 as cv2
import numpy as np
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





class stabilize():
    def __init__(self,force_camera = False,Camera_nunber = 0):
        self.previous_frame = 0
        self.frame_count = 0
        self.buffer_of_x = []

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
        plt.imshow(self.frame)
        plt.show()

    def get_shake(self,grey:np.array) ->None:
        """get shake value and will print it out

        Args:
            grey (np.array): the grey of the video
        """
        r = self.__get_metrics(grey)
        if r != None:
            print(r)

