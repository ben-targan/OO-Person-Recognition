# Ben Targan
import numpy as np
import cv2
#imutils can be found at: https://github.com/jrosebr1/imutils
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths


cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while(True):
    # Capture from video frame-by-frame
    ret, frame = cap.read()

    resizedFrame = imutils.resize(frame, width=min(400, frame.shape[1]))

    # Detect people in the resizedFrame
    # Use scale and winStride to balance speed / accuracy.
    # Larger = more speed.
    (rects, weights) = hog.detectMultiScale(resizedFrame, winStride=(8, 8), padding=(8, 8), scale=1.25)

    # Apply non-maxima suppression to the bounding boxes using a
    # Fairly large overlap threshold to try to maintain overlapping
    # Boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Draw the final bounding boxes on the resizedFrame
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(resizedFrame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the resulting resizedFrame
    cv2.imshow('Person Detection', resizedFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is finished, release the capture
cap.release()
cv2.destroyAllWindows()
