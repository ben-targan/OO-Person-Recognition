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

    frame = imutils.resize(frame, width=min(400, frame.shape[1]))
    orig = frame.copy()

    # Detect people in the frame
    # Use scale and winStride to balance speed / accuracy.
    # Larger = more speed.
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.25)

    # Draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Apply non-maxima suppression to the bounding boxes using a
    # Fairly large overlap threshold to try to maintain overlapping
    # Boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # Draw the final bounding boxes on the frame
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is finished, release the capture
cap.release()
cv2.destroyAllWindows()
