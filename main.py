import cv2
import time

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None

while True:
    check, frame = video.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = gray_frame_gau

    # delta frame is the difference between the first frame and the present frame
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)
    # threshold pixels above 60 to 255 (white)
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)
    #cv2.imshow("Video Capture", dil_frame)

    # find the limits of objects
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # iterate over contours and checks for a false positive
    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # create the rectangle from measurements
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()

