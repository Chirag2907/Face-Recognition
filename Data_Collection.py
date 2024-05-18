import cv2
import numpy as np
import os

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
data = []
offset = 50

person = input("Enter your name : ")

filename = person
path = "./data/"
ext = ".npy"

while True:
    success, frame = cam.read()
    if not success:
        print("Camera failed!")
        break

    faces = model.detectMultiScale(frame, 1.3, 2)
    faces = sorted(faces, key=lambda face: face[2]*face[3])
    if len(faces)>0:
        x,y,w,h = faces[-1]
        cv2.rectangle(frame, (x,y), (x+h, y+w), (0,255,0), 3)
        cropped_img = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        cropped_img = cv2.resize(cropped_img, (100,100))
        skip+=1
        if skip%10==0:
            data.append(cropped_img)
            print("saved so far: ", len(data))

    cv2.flip(frame, 1, frame)
    cv2.flip(cropped_img, 1, cropped_img)
    cv2.imshow('video', frame)
    cv2.imshow('cropped', cropped_img)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break

data = np.asarray(data)
data = data.reshape(data.shape[0], -1)

filepath = path + filename + ext
np.save(filepath, data)

cam.release()
cv2.destroyAllWindows()
