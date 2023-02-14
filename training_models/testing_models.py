import cv2
import numpy as np
from numpy import asarray
import pandas as pd
from PIL import Image
import os
import math
import time
import tensorflow as tf

#changing the frame into a file
def get_files(frames):
    mode=0o666
    for fraaame in frames:
            print(fraaame)
    try:
        #if folder exists, overwrite it   
        # if os.path.isdir("training_models/demo_videos") == False:
        #     os.mkdir("training_models/demo_videos", mode)

        #saves files
        np.save("training_models/demo_videos/demo_video_file", np.asarray(frames, dtype=np.float32), allow_pickle=True, fix_imports=True)
    
    except FileNotFoundError:
        pass

    return np.asarray(frames, dtype=np.float32)
        



#just give me the video path in here
# get_image(" ALY PUT THE VIDEO PATH, RUN SOME LOOP OR SOMETHING, and it should be fine.", "VIDEO URL (which will be the name of the folder)", "start_time", "endtime")

# get_image("alkfjdsl2", "test-video2", 5, 30, 29.97)

# # changing an image into an array
# def change_img(image_name):

#     img = Image.open(f"{image_name}")
 
#     # asarray() class is used to convert
#     # PIL images into NumPy arrays
#     numpydata = asarray(img)
    
#     # <class 'numpy.ndarray'>
#     print(type(numpydata))
    
#     #  shape
#     print(numpydata.shape)

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

    if i%25==0:
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

time.sleep(0.5)

# Process data
saved_classes = ['apple',
 'cheese',
 'chicken',
 'church',
 'coffee',
 'college',
 'cook',
 'daughter',
 'dog',
 'door',
 'friendly',
 'grandmother',
 'home',
 'hungry',
 'milk',
 'not know',
 'shoes',
 'thursday',
 'time',
 'tomorrow',
 'uncle',
 'vacation',
 'woman',
 'your']
X = []
file_dir = "training_models/demo_videos/demo_video_file.npy"
arr = np.load(file_dir)
for frame in arr:
    if len(X) == 0:
        X = np.array([frame])
    else:
        X = np.append(X, np.array([frame]), axis=0)
print(X.shape)

print("LOADING....")

time.sleep(0.5)

# Predict data
model = tf.keras.models.load_model('training_models/weights.03-2.05')
predictions = {n: 0 for n in saved_classes}
def predict_single_video(X):
    for frame in X:
        new_frame = tf.expand_dims(frame,0)
        print(new_frame.shape)
        preds = model.predict(new_frame)
        pred = np.argmax(preds)
        print(preds)
        pred = saved_classes[pred]
        predictions[pred] += 1
    final_prediction = max(predictions, key=predictions.get)
    return final_prediction, predictions

print(predict_single_video(X))