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



def apply_mask(face: np.array, mask: np.array) -> np.array:
    """Add the mask to the provided face, and return the face with mask.

    Args:
        face (np.array): The image around the face
        mask (np.array): The overlay onto of the face

    Returns:
        np.array: The changed image with mask overlayed on face

    """
    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = #Insert Code Here

    # Resize the mask to fit on face
    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = #Insert Code Here
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv2.resize(mask, new_mask_shape)

    # Add mask to face - ensure mask is centered
    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)
    off_h = #Insert Code Here
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w: off_w+new_mask_w][non_white_pixels] = \
         resized_mask[non_white_pixels]

    return face_with_mask


def is_out_of_frame(x0:int,y0:int,y1:int,x1:int,frame_w:int,frame_h:int) -> bool:
    if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
        return True
    else:
        return False


# load mask
mask = cv2.imread('assets/#Insert_File_name')

# initialize front face classifier
cascade = cv2.CascadeClassifier("assets/#Insert_File_name")


attempt = look_up()
frame =  attempt.run_program()
convert = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
frame = cv2.cvtColor(convert, cv2.COLOR_RGBA2RGB)
frame_h, frame_w, _ = frame.shape

# Convert to black-and-white
gray = #Insert Code Here
blackwhite = cv2.equalizeHist(gray)

# Detect faces
rects = cascade.detectMultiScale(blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE)

# Add mask to faces
for x, y, w, h in rects:
    # crop a frame slightly larger than the face
    y0, y1 = int(y - 0.25*h), int(y + 0.75*h)
    x0, x1 = x, x + w

    # give up if the cropped frame would be out-of-bounds
    if is_out_of_frame(x0,y0,y1,x1,frame_w,frame_h):
        continue

    # apply mask
    frame[y0: y1, x0: x1] = apply_mask(frame[y0: y1, x0: x1], mask)

    plt.imshow(self.frame)
    plt.show()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
