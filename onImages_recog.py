#Loading the necessary packages.
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

# Path to the tesseract for recognition.
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# The location of the file having the pre-trained EAST detector model.
net = cv2.dnn.readNet("frozen_east_text_detection.pb")


def text_detector(image):
	# Saving a original image and shape
	orig = image
	(H, W) = image.shape[:2]

	# set the new height and width to default 320
	(newW, newH) = (640, 320)

	# Calculate the ratio between original and new image for both height and weight.
	# This ratio will be used to translate bounding box location on the original image.
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the original image to new dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# Following Layers are from EAST model
	# 1. Probability scores for the region whether that contains text or not.
	# 2. Geometry of the text -- Coordinates of the bounding box detecting a text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# construct a blob from the image to forward pass it to EAST model
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)

	# Forward pass the blob from the image to get the desired output layers
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# Returns a bounding box and probability score if it is more than minimum confidence
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over rows
	for y in range(0, numRows):

		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# Find predictions and  apply non-maxima suppression
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes to find the coordinate of bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		boundary = 12
		# extract the region of interest
		text = orig[startY-boundary:endY+boundary, startX - boundary:endX + boundary]
		text = cv2.cvtColor(text.astype(np.uint8), cv2.COLOR_BGR2GRAY)
		# This will recognize the text from the image of bounding box
		textRecongized = pytesseract.image_to_string(text)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		# Putting text in image
		orig = cv2.putText(orig, textRecongized, (endX,endY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
		print(textRecongized)
	return orig

# Displaying detected image
image0 = cv2.imread('Open.jpg')
image0 = cv2.resize(image0, (640,320), interpolation = cv2.INTER_AREA)
orig = cv2.resize(image0, (640,320), interpolation = cv2.INTER_AREA)
textDetected = text_detector(image0)
cv2.imshow("Orig Image",orig)
cv2.imshow("Text Recognition", textDetected)
cv2.waitKey(0)
cv2.destroyAllWindows()