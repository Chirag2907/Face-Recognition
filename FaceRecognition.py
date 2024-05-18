import cv2
import os
import numpy as np

dataset_path = "./data/"
classID = 0
faceData = []
labels = []

for f in os.listdir(dataset_path):
    if(f.endswith(".npy")):
        # X value
        dataItem = np.load(dataset_path+f)
        faceData.append(dataItem)
        m = dataItem.shape[0]

        # Y value
        target = classID * np.ones((m,))
        classID += 1
        labels.append(target)

print(faceData)
print(labels)
