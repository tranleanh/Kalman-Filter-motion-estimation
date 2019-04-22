import cv2
import numpy as np
import time
import math

kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

def Estimate(coordX, coordY):
	# ''' This function estimates the position of the object'''
	measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
	kf.correct(measured)
	predicted = kf.predict()
	return predicted

def nothing(*arg):
	pass

if __name__ == '__main__':

	FRAME_WIDTH = 320
	FRAME_HEIGHT = 240

	# Initial HSV GUI slider values to load on program start.
	icol = (70, 65, 89, 91, 250, 255)  # Green

	cv2.namedWindow('colorTest')
	# Lower range colour sliders.
	cv2.createTrackbar('lowHue', 'colorTest', icol[0], 255, nothing)
	cv2.createTrackbar('lowSat', 'colorTest', icol[1], 255, nothing)
	cv2.createTrackbar('lowVal', 'colorTest', icol[2], 255, nothing)
	# Higher range colour sliders.
	cv2.createTrackbar('highHue', 'colorTest', icol[3], 255, nothing)
	cv2.createTrackbar('highSat', 'colorTest', icol[4], 255, nothing)
	cv2.createTrackbar('highVal', 'colorTest', icol[5], 255, nothing)

	# Load video
	video_name = 'videoplayback.mp4'

	vid = cv2.VideoCapture(video_name)
	vid.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
	vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

	ballX = 0
	ballY = 0

	while 1:

		lowHue = cv2.getTrackbarPos('lowHue', 'colorTest')
		lowSat = cv2.getTrackbarPos('lowSat', 'colorTest')
		lowVal = cv2.getTrackbarPos('lowVal', 'colorTest')
		highHue = cv2.getTrackbarPos('highHue', 'colorTest')
		highSat = cv2.getTrackbarPos('highSat', 'colorTest')
		highVal = cv2.getTrackbarPos('highVal', 'colorTest')

		_, frame = vid.read()

		frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		colorLow = np.array([lowHue, lowSat, lowVal])
		colorHigh = np.array([highHue, highSat, highVal])
		mask = cv2.inRange(frameHSV, colorLow, colorHigh)
		
		# Show the first mask
		cv2.imshow('Threshoding', mask)

		# Drawing contour
		img2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		if len(contours) != 0:
			# find the biggest area
			c = max(contours, key=cv2.contourArea)

			x, y, w, h = cv2.boundingRect(c)
			# draw the book contour (in green)
			# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			ballX = x + w // 2
			ballY = y + h // 2

		# Create Kalman Filter Object
		kfObj = cv2.KalmanFilter()
		predictedCoords = np.zeros((2, 1), np.float32)

		predictedCoords = Estimate(ballX, ballY)

		# Draw Actual coords from segmentation
		cv2.circle(frame, (ballX, ballY), 20, [0, 255, 0], 2, 8)
		cv2.line(frame, (ballX, ballY + 20), (ballX + 50, ballY + 20), [0, 255, 0], 2, 8)
		cv2.putText(frame, "CURRENT", (ballX + 50, ballY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

		# Draw Kalman Filter Predicted output
		cv2.circle(frame, (predictedCoords[0], predictedCoords[1]), 20, [0, 0, 255], 2, 8)
		cv2.line(frame, (predictedCoords[0] + 16, predictedCoords[1] - 15),
				 (predictedCoords[0] + 50, predictedCoords[1] - 30), [0, 0, 255], 2, 8)
		cv2.putText(frame, "PREDICTION", (predictedCoords[0] + 50, predictedCoords[1] - 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
		cv2.imshow('Input', frame)

		time.sleep(0.04)

		print(int(predictedCoords[0][0]),"-", int(predictedCoords[1][0]))

		k = cv2.waitKey(5) & 0XFF
		if k == 27:
			break

	vid.release()
	cv2.destroyAllWindows()