import cv2
import numpy as np
from numpy import asarray
import pandas as pd
import tensorflow as tf
import mediapipe as mp
import os
import csv
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import time
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
def extract_coordinates():
    rows = []
    #creating empty file in folder, I added the start_time in the name of the csv file, so that if a symbol appears many times in a video, it will still be created in two different csv files, just that they will have different starting times
    csv_file = f"/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/training_models/mediapipe/demo_test/demo.csv"
    # csv_file="D:/Personnel/Other learning/Programming/Personal_projects/ASL_Language_translation/training_models/mediapipe/demo_test/demo.csv"
    # if os.path.exists(csv_file):
    #     return 



# Setup CSV File for the videos
# 21 right hand landmarks, 21 left hand landmarks, 33 pose landmarks
    num_coords = 21 + 21 + 33
    landmarks = []
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    print("Initialized an empty landmarks of size:", len(landmarks))

    with open(csv_file, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)
    


#working with each video
    cap = cv2.VideoCapture(0)
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    # Read until video is completed
    else: 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while(cap.isOpened()):  
                # Capture frame-by-frame
                ret, frame = cap.read() 
                if ret == True:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False    
                    results = holistic.process(image)
                    # Display the resulting frame
                    # Right hand
                    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                            )

                    # Left Hand
                    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

                    # Pose Detections
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                            )
                    cv2.imshow('Frame',image)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    # Export coordinates
                    try:
                        # Extract Pose landmarks
                        if results.pose_landmarks:
                            pose = results.pose_landmarks.landmark
                            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        else:
                            # continue
                            pose_row=list(np.array([[0,0,0,0] for i in range(33)]).flatten())
                        # Extract hands landmarks
                        if results.right_hand_landmarks:
                            right_hand = results.right_hand_landmarks.landmark
                            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
                        else:
                            #If no right hand detected, then it writes 0 to the CSV file
                            right_hand_row = list(np.array([[0,0,0,0] for i in range(21)]).flatten())
                        if results.left_hand_landmarks:
                            left_hand = results.left_hand_landmarks.landmark
                            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
                        else:
                            #If no left hand detected, then it writes 0 to the CSV file
                            left_hand_row = list(np.array([[0,0,0,0] for i in range(21)]).flatten())

                        # Concate rows
                        row = pose_row + right_hand_row + left_hand_row
                        rows.append(row)

                        # Export to CSV
                        

                    except Exception as e:
                        print(e)
                        break
                         
                else:
                    break

            # When everything done, release the video capture object
            cap.release() 


            # Closes all the frames
            cv2.destroyAllWindows()
            with open(csv_file, mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            for row in rows:
                                csv_writer.writerow(row) 
            return results

def make_prediction(model_path, labels, csv_file):
    my_model = tf.keras.models.load_model(model_path, compile=False)
    my_model.compile(loss=SparseCategoricalCrossentropy(), optimizer=Adam(), metrics=["accuracy"])
    predictions = {n: 0 for n in labels}
    csv = pd.read_csv(csv_file)
    print(csv)
    coords = np.array(csv)
    for frame in coords:
        new_frame = tf.expand_dims(frame,0)
        preds = my_model.predict(new_frame)
        pred_value = np.argmax(preds)
        # for i in range(len(preds[0])):
        #     print(preds[0][i])
        #     predictions[saved_classes[i]] += preds[0][i]
        #     total += preds[0][i]
        predictions[labels[pred_value]] += 1
        print(preds)
    final_prediction = max(predictions, key=predictions.get)
    return predictions, final_prediction

# model_path = r"D:\Personnel\Other learning\Programming\Personal_projects\ASL_Language_translation\training_models\mediapipe\Simple-Dense-Layers\regularized-4-labels.10-0.68"# "D:/Personnel/Other learning/Programming/Personal_projects/ASL_Language_translation/training_models/mediapipe/Simple-Dense-Layers/regularized-4-labels.10-0.68"
while True:
    model_path = "/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/training_models/mediapipe/Simple-Dense-Layers/AUGMENTED-FULL-REFLECTION-WITHOUT-LEAKAGE.48-0.69"
    labels = ["coffee", "dog", "door", "milk"]
    csv_file = "/Users/aly/Documents/Programming/Apps/Machine Learning/ASL Converter/training_models/mediapipe/demo_test/demo.csv"
    input("TRY ME OUT!! ")
    extract_coordinates()
    print("LOADING...")
    time.sleep(0.5)
    print(make_prediction(model_path, labels, csv_file))
