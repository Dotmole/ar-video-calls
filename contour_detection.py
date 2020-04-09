import cv2
import matplotlib.pyplot as plt
import numpy as np

# Input image
image = cv2.imread("images/test.jpg")

# Mask
fgbg = cv2.createBackgroundSubtractorMOG2()

def contour_cropped(img: np.array) -> np.array:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
	return img

def background_subtractor_mog(img: np.array) -> np.array:
	fgmask = fgbg.apply(img)
	return fgmask

# Open a web cam
cap = cv2.VideoCapture(0)

# loop it around
while(True):
	ret, frame = cap.read()
	cv2.imshow('frame', background_subtractor_mog(frame))
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()
#plt.imshow(background_subtractor_mog(image))
#plt.show()
