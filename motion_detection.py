import cv2
import os

def motion_detection(
        frame1 : str,
        frame2 : str):
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    bboxes = []

    for contour in contours:
        if cv2.contourArea(contour) < 1500:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x,y,w,h])

    return {
        "motion_detected" : motion_detected,
        "frame" : frame1,
        "bbox" : bboxes
    }