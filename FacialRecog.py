import cv2
import numpy as np
import face_recognition


cap = cv2.VideoCapture(0)

# image to train
imgSamuel = face_recognition.load_image_file("images/samuel.9f89af64-7507-11ec-8291-00d8610a6902.jpg")
imgSamuel = cv2.cvtColor(imgSamuel, cv2.COLOR_BGR2RGB)

# test image
imgTest = face_recognition.load_image_file("images/mars-125.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# detect the face in the image
faceLoc = face_recognition.face_locations(imgSamuel)[0]
encodeSamuel = face_recognition.face_encodings(imgSamuel)[0]

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

cv2.rectangle(imgSamuel, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 0), 2)
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 0, 255), 2)

# compare the test to the ref image to check the distance between them
result = face_recognition.compare_faces([encodeSamuel], encodeTest)
print(result)

# find the distance: More the distance is smalll the result is accurate
faceDis = face_recognition.face_distance([encodeSamuel], encodeTest)
print(faceDis)

cv2.putText(imgTest, f'{result}: {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
print(encodeSamuel)
print(encodeTest)

cv2.imshow("Samuel", imgSamuel)
cv2.imshow("Samuel test", imgTest)


cv2.waitKey(0)
