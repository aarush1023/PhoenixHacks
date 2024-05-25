


import cv2
import time
import csv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         #print(0)
         return rgb_image
      else:
         #print(1)
         hand_landmarks_list = detection_result.hand_landmarks
        #  print(hand_landmarks_list,len(hand_landmarks_list))
         handedness_list = detection_result.handedness
        #  print(handedness_list,len(handedness_list))
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
           
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())

         return annotated_image
   except:
      #print(3)
      return rgb_image

def write_landmarks_to_csv(landmarks, csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Landmark", "X", "Y", "Z"])
        # Write landmarks data
        for i, landmark in enumerate(landmarks.landmark):
            writer.writerow([i, landmark.x, landmark.y, landmark.z if hasattr(landmark, 'z') else 0.0])

# Initialize MediaPipe Hand module

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # global image
    # print(result)
    global HandLandmarkerResult
    HandLandmarkerResult=result
    # print('hand landmarker result: {}'.format(result))
    #draw the landmarks on the image
    # image = draw_landmarks_on_image(output_image, result)
    # image=draw_landmarks_on_image(output_image.numpy_view(), result)
    # print(result.hand_landmarks[0])
    # print(dir(result.hand_landmarks))
    # print(result.hand_landmarks[0])
    # print(result.hand_landmarks[0].landmark[0].x,result.hand_landmarks[0].landmark[0].y)

   
 

   

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
    min_hand_presence_confidence = 0.2, # lower than value to get predictions more often
    min_tracking_confidence = 0.3,
    result_callback=print_result)

cap=cv2.VideoCapture(0)
landmarker=HandLandmarker.create_from_options(options)
while True:
    success, image=cap.read()
    if not success:
        #print("Ignoring empty camera frame.")
        continue
    temp_image = cv2.flip(image, 1)
    # with HandLandmarker.create_from_options(options) as landmarker:
        # flip the image
       
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=temp_image)
    landmarkerResults=landmarker.detect_async(image=mp_image, timestamp_ms=int(time.time() * 1000))
    temp_image=draw_landmarks_on_image(temp_image,HandLandmarkerResult)
    #print(HandLandmarkerResult)
       

   
    cv2.imshow('MediaPipe Hands',temp_image)
    if cv2.waitKey(1) == ord('q'):
        break


