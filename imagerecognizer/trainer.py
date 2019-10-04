import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def ImageId(path):
    imgpaths=[os.path.join(path,f) for f in os.listdir(path)]
    face=[]
    IDs=[]
    for imgpath in imgpaths:
        #print(imgpaths)
        faceimg=Image.open(imgpath).convert("L")
        facenp = np.array(faceimg , 'uint8')
        ID=int(os.path.split(imgpath)[-1].split('.')[1])
        face.append(facenp)
        print(ID)
        IDs.append(ID)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return IDs,face

Ids,face = ImageId(path)
recognizer.train(face,np.array(Ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()