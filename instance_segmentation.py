import os
import time
import cv2


# Initializing paths
dataPath = 'mask-rcnn-coco'
labelsPath = os.path.sep.join([dataPath, "object_detection_classes_coco.txt"])
configPath = os.path.sep.join([dataPath, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
weightsPath = os.path.sep.join([dataPath, "frozen_inference_graph.pb"])

# Labels on which the model was trained
LABELS = open(labelsPath).read().strip().split("\n")

# Loading Mask R-CNN trained data
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# Constructing the kernal
K = (41, 41)

# Initializing video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop it around
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

	idxs = np.argsort(boxes[0, 0, :, 2])[::-1]

	mask = None
	roi = None
	coords = None

	# Loop over the indices
	# Find the prediction which is person
	# Extract the person image
	# Add it to the front view stream coming from the first user
	# Display it

