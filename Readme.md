## Project Face Recogntion

#### Description
This project is a face recognition project. It is a simple project that uses opencv for collecting the data.
The model is trained using the collected data and then predictions are made using KNN algorithm.

#### Steps

###### 1. Collect data of various people.
 - asking multiple people to come in front of the webcam, then taking 10-20 images of each person.
 - The images are then stored in a folder with the name of the person.
- Only the face of the person is stored. This is done using haarcascade classifier.

###### 2. Train the model
 - The images are then converted to numpy arrays and stored in a list.
 - The labels are also stored in a list.
 - The model is then trained using the data.

###### 3. Predictions
- The model is then used to predict the person in front of the camera.
- The model uses KNN algorithm to predict the person.
