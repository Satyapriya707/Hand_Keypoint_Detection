import cv2
import time
import mediapipe as mp

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Initializing the drawng utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils


# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)
 
# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
 
import matplotlib.pyplot as plt
import os
import json
out_dir = "2_out"
c = 0

while capture.isOpened():
    # capture frame by frame
    ret, frame = capture.read()
 
    # resizing the frame for better view
    frame = cv2.resize(frame, (800, 600))
 
    # Converting the from from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    if results.right_hand_landmarks:
      landmarks=results.right_hand_landmarks.landmark # landmarks[0].x
      lm = {}
      lm["d1"] = ((landmarks[8].x - landmarks[12].x)**2 + (landmarks[8].y - landmarks[12].y)**2)**0.5
      lm["d2"] = ((landmarks[12].x - landmarks[16].x)**2 + (landmarks[12].y - landmarks[16].y)**2)**0.5
      lm["d3"] = ((landmarks[8].x - landmarks[16].x)**2 + (landmarks[8].y - landmarks[16].y)**2)**0.5
      # lm = []
      # for i in range(len(landmarks)):
      #   lm.append((i, landmarks[i].x, landmarks[i].y))
      with open(os.path.join(out_dir, f"{c}.json"), "w") as f:
        json.dump(lm, f)
      # plt.imsave(os.path.join(out_dir, f"{c}.jpg"), image)
      # c += 1

 
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # # Drawing the Facial Landmarks
    # mp_drawing.draw_landmarks(
    #   image,
    #   results.face_landmarks,
    #   mp_holistic.FACEMESH_TESSELATION,
    #   mp_drawing.DrawingSpec(
    #     color=(255,0,255),
    #     thickness=1,
    #     circle_radius=1
    #   ),
    #   mp_drawing.DrawingSpec(
    #     color=(0,255,255),
    #     thickness=1,
    #     circle_radius=1
    #   )
    # )
 
    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
      image,
      results.right_hand_landmarks,
      mp_holistic.HAND_CONNECTIONS
    )
 
    # # Drawing Left hand Land Marks
    # mp_drawing.draw_landmarks(
    #   image,
    #   results.left_hand_landmarks,
    #   mp_holistic.HAND_CONNECTIONS
    # )

    # # Code to access landmarks
    # for landmark in mp_holistic.HandLandmark:
    #     print(landmark, landmark.value)
    
    # print(mp_holistic.HandLandmark.WRIST.value)
     
    # Calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime
     
    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
  
    plt.imsave(os.path.join(out_dir, f"{c}.jpg"), image)
    c += 1
    # Display the resulting image
    cv2.imshow("Facial and Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()