import cv2
import numpy as np
import face_recognition
from timeit import default_timer as timer
from datetime import datetime
from datetime import timedelta
import time
import os
import dlib.cuda as cuda
print(cuda.get_num_devices())




def FindEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def MarkFace(name):
    with open("RecordDetection.csv","r+") as f:
        myDateList = f.readlines()
        nameList = []
        for line in myDateList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{dtString}")

def main():
    path = "Images"
    imagesRef = []
    classNames = []
    myList = os.listdir(path)

    for cls in myList:
        curImg = cv2.imread(f"{path}/{cls}")
        imagesRef.append(curImg)
        classNames.append(os.path.splitext(cls)[0])

    encodeListKnown = FindEncodings(imagesRef)
    print(f"Encoding Complete ==> {len(encodeListKnown)} reference faces")

    cap = cv2.VideoCapture(0)
    prevTime = 0
    CurTime = 0

    while True:
        scale = 4.0
        success, img = cap.read()
        # to speed up the process and increase the fps you may reduce hte size of the image
        imgS = cv2.resize(img, (0,0),None, 1/scale,1/scale )
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # convert to RGB

        start = timer()
        # first we need to collect hte location o fall the faces
        facesFrame = face_recognition.face_locations(imgS)
        encodeFrame = face_recognition.face_encodings(imgS,facesFrame)
        end = timer()
        print('Time to detect', timedelta(seconds=end - start))

        # Compare the faces found with the knowned faces we loaded
        for encodeFace, faceLoc in zip(encodeFrame, facesFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
            # will get list with the face distance compare to all the know faces
            # print(faceDist)
            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                name = classNames[matchIndex]
                MarkFace(name)
                # print(name)
                y1,x2,y2,x1 = faceLoc # remap to the unscaled
                y1, x2, y2, x1 = y1*int(scale), x2*int(scale), y2*int(scale), x1*int(scale)
                cv2.rectangle(img, (x1, y1), (x2, y2), (56, 204, 231), 2)
                cv2.rectangle(img, (x1, y1+35), (x2, y1), (56, 204, 231), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y1+28), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


        # fps management to check performance
        CurTime = time.time()
        fps = 1 / (CurTime - prevTime)
        prevTime = CurTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("WebCam Feed", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
