import cv2
import cv2
import numpy as np
from numpy import asarray
import pandas as pd
from PIL import Image
import os
import math

#changing the frame into a file
def get_files(frames):
    mode=0o666
    for fraaame in frames:
            print(fraaame)
    try:
        #if folder exists, overwrite it   
        # if os.path.isdir("demo_videos/") == False:
        #     os.mkdir("demo_videos/", mode)

        #saves files
        np.save("demo_video_file", np.asarray(frames, dtype=np.float32), allow_pickle=True, fix_imports=True)
    
    except FileNotFoundError:
        pass

    return np.asarray(frames, dtype=np.float32)
        



#just give me the video path in here
# get_image(" ALY PUT THE VIDEO PATH, RUN SOME LOOP OR SOMETHING, and it should be fine.", "VIDEO URL (which will be the name of the folder)", "start_time", "endtime")

# get_image("alkfjdsl2", "test-video2", 5, 30, 29.97)

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

cam = cv2.VideoCapture(0)

i=0
frames=[]
while True:
    check, frame = cam.read()
    # print(frame)

    #inputting the image into the frames list.
    img=frame

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

    if i%15==0:
        get_files(frames)
    
    i+=1
    cv2.imshow('video', frame)


    key = cv2.waitKey(1)
    #breaks if we write escape, escape is key 27
    if key == 27:
        break
    # breaK

cam.release()
cv2.destroyAllWindows()