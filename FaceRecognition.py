import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# Importing dataset
dataset_path = "./data/"
classID = 0
faceData = []
labels = []
mp = {}

for f in os.listdir(dataset_path):
    if(f.endswith(".npy")):
        mp[classID] = f[:-4]
        # X value
        dataItem = np.load(dataset_path+f)
        faceData.append(dataItem)
        m = dataItem.shape[0]

        # Y value
        target = classID * np.ones((m,))
        classID += 1
        labels.append(target)

X_train = np.concatenate(faceData, axis=0)
y_train = np.concatenate(labels, axis=0)

# KNN Model Training
# model = KNeighborsClassifier(n_neighbors=20, metric='minkowski', p=2)
# model.fit(X_train, y_train)

# SVM Model Training
# model = svm.LinearSVC(dual='auto')
# model.fit(X_train, y_train)

# Random Forest Model Training
model = RandomForestClassifier(n_estimators=200, criterion='log_loss', min_samples_split=20)
model.fit(X_train, y_train)

# Prediction!
cam = cv2.VideoCapture(0)
classifer = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
offset = 50

while True:
    success, frame = cam.read()
    if not success:
        print("Camera failed!")
        break
    
    faces = classifer.detectMultiScale(frame, 1.3, 2)
    if(len(faces)>0):
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
            try:
                cropped_img = frame[y-offset:y+h+offset, x-offset:x+w+offset]
                cropped_img = cv2.resize(cropped_img, (100,100))
                cropped_img = cropped_img.reshape(1,-1)
                prediction = model.predict(cropped_img)
                namepredicted = mp[int(prediction)]
                cv2.putText(frame, namepredicted, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            except Exception as e:
                print(str(e))
            
    cv2.imshow('capture',frame)
    key = cv2.waitKey(1)
    if key==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

