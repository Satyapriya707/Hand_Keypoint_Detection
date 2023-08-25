import cv2
import time
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import json
import tkinter
import mouse
import numpy as np

from config import minVal, maxVal, scr_height_factor, scr_width_factor, mouse_speed, click_wait_time
from mouse_utils import check_activity, do_activity

out_dir = "4_out"


root = tkinter.Tk()
root.withdraw()
scr_width, scr_height = root.winfo_screenwidth(), root.winfo_screenheight() # w - 1536, h - 864

mp_hand = mp.solutions.hands
hand_model = mp_hand.Hands(
    static_image_mode=False,
    model_complexity=1, 
    # smooth_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0
 
c = 0

os.makedirs(out_dir, exist_ok=True)

last_frame_landmark = [0, 0, 0, 0, 0, 0]

l_cl_flag = True
r_cl_flag = True
start_flag = True

out_val = ""

while capture.isOpened():
    ret, frame = capture.read()

    frame = cv2.resize(frame, (int(scr_width/scr_width_factor), int(scr_height/scr_height_factor)))
    # frame = cv2.resize(frame, (800, 600))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hand_model.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
      landmarks=results.multi_hand_landmarks[0].landmark

      # lm = {}
      # lm["d1"] = ((landmarks[8].x - landmarks[12].x)**2 + (landmarks[8].y - landmarks[12].y)**2)**0.5
      # lm["d2"] = ((landmarks[12].x - landmarks[16].x)**2 + (landmarks[12].y - landmarks[16].y)**2)**0.5
      # lm["d3"] = ((landmarks[8].x - landmarks[16].x)**2 + (landmarks[8].y - landmarks[16].y)**2)**0.5

      # lm["val"] = []
      # for i in range(len(landmarks)):
      #   lm["val"].append((i, landmarks[i].x, landmarks[i].y, landmarks[i].z))

      # with open(os.path.join(out_dir, f"{c}.json"), "w") as f:
      #   json.dump(lm, f)

      z_val = []
      for i in [8,12,16]:
        z_val.append(landmarks[i].z)
      norm_z = np.array(z_val)
      new_norm_z = (norm_z - np.mean(norm_z))/np.std(norm_z)
      d1, d2 = abs((new_norm_z[0] - new_norm_z[1])/new_norm_z[1]), abs((new_norm_z[0] - new_norm_z[2])/new_norm_z[1])
      norm_z = np.append(np.array([c+1]), norm_z)
      new_norm_z = np.append(norm_z, new_norm_z)
      out_val += str(np.append(new_norm_z, [d1, d2])) + "\n"

    ######################## logic ###########################
    if not start_flag:
      if results.multi_hand_landmarks:
        act = check_activity(lm, minVal, maxVal)
        if act == "left_click":
          if not r_cl_flag:
            r_cl_flag = True
          if l_cl_flag:
            l_cl_start = time.time()
            l_cl_flag = False
          if time.time() - l_cl_start >= click_wait_time:
            do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed)
            l_cl_flag = True
        elif act == "right_click":
          if not l_cl_flag:
            l_cl_flag = True
          if r_cl_flag:
            r_cl_start = time.time()
            r_cl_flag = False
          if time.time() - r_cl_start >= click_wait_time:
            do_activity(act, landmarks[12].x, landmarks[12].y, last_frame_landmark[2], last_frame_landmark[3], scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed)
            r_cl_flag = True
        elif act == "move":
          if not r_cl_flag:
            r_cl_flag = True
          if not l_cl_flag:
            l_cl_flag = True
          # do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], scr_width_factor, scr_height_factor, scr_width, scr_height, time.time()-previousTime)
          do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed)
        elif act == "drag":
          if not r_cl_flag:
            r_cl_flag = True
          if not l_cl_flag:
            l_cl_flag = True
          # do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], scr_width_factor, scr_height_factor, scr_width, scr_height, time.time()-previousTime)
          do_activity(act, landmarks[8].x, landmarks[8].y, last_frame_landmark[0], last_frame_landmark[1], scr_width_factor, scr_height_factor, scr_width, scr_height, mouse_speed)

    ######################## logic ###########################
    
    if results.multi_hand_landmarks:
      # start_flag = False  ############################## SET TO ALWAYS TRUE TO STOP CURSOR MOVEMENT ############################
      last_frame_landmark = [landmarks[8].x, landmarks[8].y, landmarks[12].x, landmarks[12].y, landmarks[16].x, landmarks[16].y]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        image,
        results.multi_hand_landmarks[0],
        mp_hand.HAND_CONNECTIONS
      )
 
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

    if results.multi_hand_landmarks:
      plt.imsave(os.path.join(out_dir, f"{c}.jpg"), image)
      c += 1

    cv2.imshow("Facial and Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

with open(os.path.join(out_dir, "abc.txt"), "w") as f:
  f.write(out_val)

capture.release()
cv2.destroyAllWindows()