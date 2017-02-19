""" Experiment with face detection and image filtering using OpenCV """

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((21,21),'uint8')

while True:
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :],kernel)
        cv2.circle(frame,(int(x+w/3),int(y+h/3)),int(h/8),(255, 255, 255),-1)
        cv2.circle(frame,(int(x+w*2/3),int(y+h/3)),int(h/8),(255, 255, 255),-1)
        cv2.circle(frame,(int(x+w/3),int(y+h/3)+10),int(h/16),(0, 0, 0),-1)
        cv2.circle(frame,(int(x+w*2/3),int(y+h/3)-10),int(h/16),(0, 0, 0),-1)
        cv2.ellipse(frame,(int(x+w/2),int(y+h)),(int(w/2),int(h/3)),0,225,315,(255,0,0),3)
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))

    # Display the resulting framep
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
