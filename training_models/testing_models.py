import cv2
import cv2
import numpy as np
from numpy import asarray
import pandas as pd
from PIL import Image
import os
import math
def get_image(video_url, video_path, start_time, end_time, fps):
    #get the right path
    capture = cv2.VideoCapture(f"training_video_data/{video_url}.mp4")
    frameNr = 0
    start_frame=start_time*fps
    end_frame=end_time*fps

    # buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    frames=[]
    while (True):
        success, img = capture.read()
        # print(frameNr)
        if success and frameNr>start_frame and frameNr<end_frame:
            if frameNr%30==0: 
                #cropping the image
             


                # #transform into numpy array
                w, h, c = img.shape

                if w < 226 or h < 226:
                    d = 226. - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

                if w > 256 or h > 256:
                    img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

                img = (img / 255.) * 2 - 1

                frames.append(img)

                #this line used to write it into a folder
                # cv2.imwrite(f'{video_url}/frame_{frameNr}.jpg', img)

        #end of the video, or end of the portion of the video
        if frameNr>end_frame:
            break
    
        frameNr = frameNr+1

    capture.release()



    return np.asarray(frames, dtype=np.float32)
        



#just give me the video path in here
# get_image(" ALY PUT THE VIDEO PATH, RUN SOME LOOP OR SOMETHING, and it should be fine.", "VIDEO URL (which will be the name of the folder)", "start_time", "endtime")

# get_image("alkfjdsl2", "test-video2", 5, 30, 29.97)

cam = cv2.VideoCapture(0)
# changing an image into an array
def change_img(image_name):
    img = Image.open(f"{image_name}")
 
    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = asarray(img)
    
    # <class 'numpy.ndarray'>
    print(type(numpydata))
    
    #  shape
    print(numpydata.shape)


while True:
    check, frame = cam.read()
    print(frame)
    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()