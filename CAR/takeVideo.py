import cv2
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

import board
import digitalio
from imutils.video import VideoStream

import board
import digitalio
import time


button = digitalio.DigitalInOut(board.D18)
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.UP

button1 = digitalio.DigitalInOut(board.D4)
button1.direction = digitalio.Direction.INPUT
button1.pull = digitalio.Pull.UP


def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=20,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor_mode=0 !"
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# vs = VideoStream(src=gstreamer_pipeline(flip_method=0)).start()
# vs.stream.release()
# vs.stop()
vs = VideoStream(src=gstreamer_pipeline(flip_method=0)).start()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data3.avi',fourcc, 20.0, (640,360))

record = True

while(True):
    # if (not button.value):
    #     record = True
    
    if (record):
        frame = vs.read()
        # vs.stream.release()
        # vs.stop()
        frame = cv2.flip(frame, -1)
        out.write(frame)
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
    if c== ord('q'):
        break

vs.stream.release()
vs.stop()
out.release()
cv2.destroyAllWindows()