from stablizer import stabilize
import cv2.cv2 as cv2


extras = stabilize()
while True:
    frame = extras.get_video_frame()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    extras.get_shake(grey)
    extras.show_image()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break