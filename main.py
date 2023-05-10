from datetime import time

import cv2

# Load the video
cap = cv2.VideoCapture('poissons3.mp4')

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Find contours of moving objects
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours and select only fish contours
    fish_contours = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 150:  # adjust threshold as needed
            fish_contours.append(cnt)

    # Object tracking
    for cnt in fish_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw lines connecting the centers of the rectangles
    for i in range(len(fish_contours)-1):
        cnt1 = fish_contours[i]
        cnt2 = fish_contours[i+1]
        x1, y1, w1, h1 = cv2.boundingRect(cnt1)
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)
        center_x1 = x1 + w1 // 2
        center_y1 = y1 + h1 // 2
        center_x2 = x2 + w2 // 2
        center_y2 = y2 + h2 // 2
        cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
