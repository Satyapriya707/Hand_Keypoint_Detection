import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import json
import tkinter
import mouse
import numpy as np

from config import minVal, maxVal, scr_height_factor, scr_width_factor, mouse_speed, \
  click_wait_time, click_sleep_time, minPixel, depth_multiply_factor
from mouse_utils import check_activity, do_activity, if_pause

root = tkinter.Tk()
root.withdraw()
scr_width, scr_height = root.winfo_screenwidth(), root.winfo_screenheight() # w - 1536, h - 864

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0
 
# out_dir = "2_out"
# c = 0

last_frame_landmark = [0, 0, 0, 0, 0, 0, 0, 0, 0]

l_cl_flag = True
r_cl_flag = True
start_flag = True

while capture.isOpened():
    ret, frame = capture.read()

    frame = cv2.resize(frame, (int(scr_width/scr_width_factor), int(scr_height/scr_height_factor)))
    # frame = cv2.resize(frame, (800, 600))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    if results.right_hand_landmarks:
      landmarks=results.right_hand_landmarks.landmark
      lm = {}
      lm["d1"] = ((landmarks[8].x - landmarks[12].x)**2 + (landmarks[8].y - landmarks[12].y)**2)**0.5
      lm["d2"] = ((landmarks[12].x - landmarks[16].x)**2 + (landmarks[12].y - landmarks[16].y)**2)**0.5
      lm["d3"] = ((landmarks[8].x - landmarks[16].x)**2 + (landmarks[8].y - landmarks[16].y)**2)**0.5

      # with open(os.path.join(out_dir, f"{c}.json"), "w") as f:
      #   json.dump(lm, f)

    ######################## logic ###########################
    if not start_flag:
      if results.right_hand_landmarks and not if_pause(lm, landmarks[4].x, landmarks[4].y, landmarks[8].x, \
        landmarks[8].y, minVal, maxVal):
        th = []
        for i in [8,12,16]:
          d =  ((landmarks[i].x - landmarks[4].x)**2 + (landmarks[i].y - landmarks[4].y)**2)**0.5
          th.append(d)
        act = check_activity(lm, minVal, maxVal, landmarks[8].z, landmarks[12].z, landmarks[16].z,\
          landmarks[4].z, th, depth_multiply_factor)
        if act == "left_click":
          if not r_cl_flag:
            r_cl_flag = True
          if l_cl_flag:
            l_cl_start = time.time()
            l_cl_flag = False
          if time.time() - l_cl_start >= click_wait_time:
            do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], \
              scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed, click_sleep_time)
            l_cl_flag = True
        elif act == "right_click":
          if not l_cl_flag:
            l_cl_flag = True
          if r_cl_flag:
            r_cl_start = time.time()
            r_cl_flag = False
          if time.time() - r_cl_start >= click_wait_time:
            do_activity(act, landmarks[12].x, landmarks[12].y, last_frame_landmark[3], last_frame_landmark[4], \
              scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed, click_sleep_time)
            r_cl_flag = True
        elif act == "move":
          if not r_cl_flag:
            r_cl_flag = True
          if not l_cl_flag:
            l_cl_flag = True
          do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], \
            scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed, click_sleep_time, minPixel)
        elif act == "drag":
          if not r_cl_flag:
            r_cl_flag = True
          if not l_cl_flag:
            l_cl_flag = True
          do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], \
            scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed, click_sleep_time, minPixel)
    else:
      cv2.namedWindow("Hand Landmarks")
      cv2.setWindowProperty("Hand Landmarks", cv2.WND_PROP_TOPMOST, 1)
      cv2.moveWindow("Hand Landmarks", -15,scr_height-int(scr_height/scr_height_factor)-20)

    ######################## logic ###########################
    
    if results.right_hand_landmarks:
      start_flag = False
      last_frame_landmark = [landmarks[8].x, landmarks[8].y, landmarks[8].z, landmarks[12].x, landmarks[12].y, landmarks[12].z, landmarks[16].x, landmarks[16].y, landmarks[16].z]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.right_hand_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
      )
 
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    # if results.right_hand_landmarks:
    #   plt.imsave(os.path.join(out_dir, f"{c}.jpg"), image)
    #   c += 1

    cv2.imshow("Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()