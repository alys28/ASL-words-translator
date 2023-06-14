#https://www.analyticsvidhya.com/blog/2022/03/background-removal-in-the-image-using-the-mediapipe-library/
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from PIL import Image
import os
import math


# Define the video capture object
cap = cv2.VideoCapture('test-video.mp4')

# # Define the mediapipe solutions for face detection and segmentation
# mp_face_detection = mp.solutions.face_detection
# mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Loop through each frame of the video
frame_num = 0
mx_number=5
while cap.isOpened():
    # Read the frame from the video
    success, frame = cap.read()
    # print(frame.shape)
    if not success or frame_num>mx_number:
        break
    cv2.imshow('Frame', frame)

    print(frame_num)

    #displaying the image

    # # Convert the frame to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # # Detect the face in the frame
    # with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    #     results = face_detection.process(frame)
        
    #     if results.detections:
    #         # Get the face bounding box
    #         bbox = results.detections[0].location_data.relative_bounding_box
    #         height, width, channels = frame.shape
            
    #         # Convert the bounding box to pixel coordinates
    #         xmin = int(bbox.xmin * width)
    #         ymin = int(bbox.ymin * height)
    #         w = int(bbox.width * width)
    #         h = int(bbox.height * height)
            
    #         # Apply the selfie segmentation to the frame
    #         with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    #             results = selfie_segmentation.process(frame)
                
    #             # Create the mask by applying the segmentation mask to the face bounding box
    #             mask = cv2.resize(results.segmentation_mask, (w, h))
    #             mask = (mask > 0.1).astype('uint8') * 255
                
    #             # Apply the mask to the frame
    #             frame = frame[ymin:ymin+h, xmin:xmin+w]
    #             mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #             frame = cv2.bitwise_and(frame, mask)
                
    #             # Resize the frame to the original size
    #             frame = cv2.resize(frame, (width, height))
                
    # Save the frame as an image
    cv2.imwrite(f'output/frame{frame_num:04d}.jpg', frame)

    frame_num += 1
    
   

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()



#saving numpy array
mode=0o666

# try:
#     #if file doesn't exist, create it   
#     if os.path.isdir(f"output/{video_path}") == False:
#         os.mkdir(f"output/{video_path}", mode)
#     np.save(f"train_data/{video_path}/{video_url}", np.asarray(frames, dtype=np.float32), allow_pickle=True, fix_imports=True)

# except FileNotFoundError:
#     pass
