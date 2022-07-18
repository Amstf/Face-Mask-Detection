# import the necessary packages
from logging import warning
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils

from imutils.video import VideoStream
import numpy as np
import time
import cv2
import os
threshold = 220
# output formatting
cap = cv2.VideoCapture('vid4.mp4')
output_filename = 'detected1.mp4'
output_frames_per_second = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
file_size = (1280, 720)
result = cv2.VideoWriter(output_filename,fourcc,output_frames_per_second,file_size)
def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0,
		mean = (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []
	centers = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			
			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			# cv2.circle(frame, (startY + int(endY * 0.5), startX +int(endX * 0.5)), 4, (0, 255, 0), -1)
			cv2.circle(frame,(getCenter(startX, startY, endX, endY)), 4, (135,206,235), -1)
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			# [(left+right)//2, (top+bottom)//2]

			
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			centers.append(getCenter(startX, startY, endX, endY))
			
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)


	# return a 2-tuple of the face locations and their corresponding
	# locations

	return (locs, preds),centers

# load our serialized face detector model from disk
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream




#function that return the Center of an object
def getCenter(startX, startY, endX, endY):
	
    return [(startX+endX)//2, (startY+endY)//2]

# loop over the frames from the video stream
print("[INFO] Reading video stream...")
while cap.isOpened():
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=400)
    if ret:
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
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
        img=cv2.resize(frame, file_size)
        result.write(img)

    else:
        break
        
          
cap.release()
result.release()


