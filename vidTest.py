import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1920)


# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv.imshow("Frame", frame)

        # Get the frame rate of the camera
        fps = cap.get(cv.CAP_PROP_FPS)
        print(f"FPS of the webcam: {fps}")

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord("q"):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
