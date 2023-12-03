# import the necessary packages

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np

import cv2
cv2.startWindowThread()

threshold = 320
path="1.jpeg"
frame = cv2.imread(path)
def detect_and_predict_mask(frame, faceNet, maskNet):

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0,mean= (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
	
    faces = []
    locs = []
	
    preds = []
	
    centers = []

   # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            cv2.circle(frame,(getCenter(startX, startY, endX, endY)), 4, (135,206,235), -1)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
	    # add the face and bounding boxes to their respective	
            centers.append(getCenter(startX, startY, endX, endY))	
            faces.append(face)	
            locs.append((startX, startY, endX, endY))

	
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds),centers

prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")
#function that return the Center of an object
def getCenter(startX, startY, endX, endY):
    return [(startX+endX)//2, (startY+endY)//2]

# loop over the frames from the video stream
print("[INFO] Reading video stream...")
(locs, preds),centers = detect_and_predict_mask(frame, faceNet, maskNet)
        # distances_list
distances_list = np.empty((len(centers), len(centers)))
        # loop over the detected face locations and their corresponding
        # locations
for i in range(len(centers)):
        for j in range(len(centers)):
            distance = np.linalg.norm(np.array(centers[i])-np.array(centers[j]))
            if distance==0:
                distances_list[i,j] = np.inf
            else:  
                distances_list[i,j] = distance
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # extract the minimum distance between each person
            min_distances = distances_list.min(axis=0)
            for face in range(len(centers)):
                if min_distances[face]<threshold:
                    text= "warning"
                    cv2.putText(frame, text, (endX, endY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # # determine the class label and color we'll use to draw
            # # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.imwrite("de.jpeg", frame)
            cv2.imshow('frame',frame)

        



