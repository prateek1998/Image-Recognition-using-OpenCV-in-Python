import cv2
import numpy as np
facedetect = cv2.CascadeClassifier('classifier\haarcascade_frontalface_default.xml')
eyedetect = cv2.CascadeClassifier('classifier\haarcascade_eye.xml')
cap=cv2.VideoCapture(0)

id = input('enter user id')
i=0
while True:
    res,img=cap.read()
    img=cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        i+=1
        cv2.imwrite("dataset/user."+str(id)+"."+str(i)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x + w]
        roi_img = img[y:y + h, x:x + w]
        eyes = eyedetect.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow("win",img)
    cv2.waitKey(1)
    if i>50:
        break
cap.release()
cv2.destroyAllWindows()
