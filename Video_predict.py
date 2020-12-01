'''
# INSTALL FOLLOWING LIBRARIES
---------------------------------------------------------------------
1. OPENCV --> Linux - sudo apt-get install python-opencv
	  --> Python - pip install opencv-python
---------------------------------------------------------------------
2. NUMPY --> Linux - sudo apt-get install python-numpy python-scipy
	 --> Python - pip install numpy
---------------------------------------------------------------------
3. TENSORFLOW --> Python - pip install tensorflow
---------------------------------------------------------------------
4. KERAS
---------------------------------------------------------------------
'''

import cv2
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('CNN_1.model') # USE YOUR PATH TO THE MODEL

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def nothing(x):
    pass

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 5, 20, nothing)

capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    if not _:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")
    
    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    
    for rect in faces:
        (x, y, w, h) = rect
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0),2)
        roi_gray = gray[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(80, 80))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        
        prediction = model.predict(img_pixels)
        
        max_index = np.argmax(prediction[0])
        
        emotions = ('Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise') 
        prediction_emotion = emotions[max_index]
        
        cv2.putText(frame, prediction_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        resized_img = cv2.resize(frame, (700, 500))
        #cv2.imshow('Facial emotion analysis ',resized_img)
        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break
    
capture.release()
cv2.destroyAllWindows()
