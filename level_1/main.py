from face_activity import face_activity
import cv2


camera = face_activity(True,2)

while True:
    frame = camera.get_video_frame()
    grey = camera.convert_to_grey(frame)
    people = camera.detect_human(grey)
    
    for x,y,w,h in people:
        camera.crop_image(x,y,w,h)
        if camera.is_out_of_frame():
            continue

        camera.apply_mask()
    
    camera.show_image(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.end_program()