import cv2
import numpy as np
from PIL import Image
facedetect = cv2.CascadeClassifier('classifier\haarcascade_frontalface_default.xml')
eyedetect = cv2.CascadeClassifier('classifier\haarcascade_eye.xml')
cap=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\trainingdata.yml')
id=0
font = cv2.FONT_HERSHEY_DUPLEX

while True:
    res,img=cap.read()
    img=cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        eyes = eyedetect.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_img,(ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            id,conf=rec.predict(gray[y:y+h,x:x+w])
            if (id==1) or (id==6):
                id = "prateek"
            elif (id == 7):
                id = "elon musk"
            elif (id == 3):
                id = "narendra modi"
            elif (id == 8):
                id = "harsh"
            cv2.putText(img, str(id), (x, y + h), font, 1, (0, 255, 0), 2)
            '''elif (id==2):
                id = "priyanka"
            elif (id == 3):
                id = "nikhil"
            elif (id == 4):
                id = "prabhat"
            elif (id == 5):
                id = "neelam"
            '''
            #cv2.putText(img,str(id),(x,y+h),font,1,(0,255,0),2)

    cv2.imshow("win",img)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
