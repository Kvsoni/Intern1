from ultralytics import YOLO as yolo
import cv2 as cv
import numpy as np

## Function to detect the difference between two frames.
def diff(prev, frame):
    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Compute the Mean Squared Error (MSE)
    mse = ((prev - frame) ** 2).mean()
    return mse

## Function for angle calculation
def angle_calc(p1: list, p2: list, p3: list):
    if (p1.all() == 0) or (p2.all() == 0) or (p3.all() == 0):
        return -1
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = (np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # print("p1",p1,"p2",p2,"p3",p3,"v1",v1,"v2",v2,"Cos_theta",cos_theta)
    theta = np.arccos(cos_theta)  # angle in radians
    angle = abs(theta * 180.0 / np.pi)  # angle in degree
    if angle > 180:
        angle = 360 - angle
    return int(angle)

## Calculating 8 important angles for pose detection.
### This function is specific to tis code application.
def joint_angles(joint: list):
    joints = np.zeros((17, 2))
    joints[: joint.shape[0], : joint.shape[1]] = joint
    out = []
    out.append(angle_calc(joints[9], joints[7], joints[5]))  ### Angle 1 on right side
    out.append(angle_calc(joints[10], joints[8], joints[6]))  ### Angle 2 on left
    out.append(angle_calc(joints[7], joints[5], joints[11]))  ### Angle 3 on right
    out.append(angle_calc(joints[8], joints[6], joints[12]))  ### Angle 4 on left
    out.append(angle_calc(joints[5], joints[11], joints[13]))  ### Angle 5 on right
    out.append(angle_calc(joints[6], joints[12], joints[14]))  ### Angle 6 on left
    out.append(angle_calc(joints[11], joints[13], joints[15]))  ### Angle 7 on right
    out.append(angle_calc(joints[12], joints[14], joints[16]))  ### Angle 8 on left
    return out

## Special Cases for this code to determine the pose names.
def pose(flexion_angles: list):
    l = len(flexion_angles)
    up = flexion_angles[: int(l / 2)]
    down = flexion_angles[int(l / 2) :]
    final = []
    text = 0

    def is_between(value, min_val, max_val):
        return all(min_val[i] <= value[i] <= max_val[i] for i in range(len(value)))

    match up:  ## Hands Up condition
        case _ if is_between(up, [0, 0, 160, 160], [0, 0, 180, 180]):
            text = "Both Hands Up"
        case _ if is_between(up, [0, 0, 160, 0], [0, 0, 180, 0]):
            text = "Right Hand up"
        case _ if is_between(up, [0, 0, 0, 160], [0, 0, 0, 180]):
            text = "Left Hand Up"

    if text != 0:
        final.append(text)
        text = 0

    match up:  ## Hands Raised condition
        case _ if is_between(up, [80, 80, 80, 80], [100, 100, 100, 100]):
            text = "Both Hands Raised Up"
        case _ if is_between(up, [80, 0, 80, 0], [100, 0, 100, 0]):
            text = "Right Hand Raised up"
        case _ if is_between(up, [0, 80, 0, 80], [0, 100, 0, 100]):
            text = "Left Hand Raised Up"

    if text != 0:
        final.append(text)
        text = 0

    match up:  ## Hands Horizontal condition
        case _ if is_between(up, [160, 160, 80, 80], [180, 180, 100, 100]):
            text = "Both hands are horizontal"
        case _ if is_between(up, [160, 160, 0, 80], [180, 180, 20, 100]):
            text = "Right hand is horizontal"
        case _ if is_between(up, [160, 160, 80, 0], [180, 180, 100, 20]):
            text = "Left hand is horizontal"

    if text != 0:
        final.append(text)
        text = 0

    match up:  ## Hands Down condition
        case _ if is_between(up, [130, 130, 0, 0], [180, 180, 30, 30]):
            text = "Both Hands Down"
        case _ if is_between(up, [0, 130, 0, 0], [0, 180, 30, 30]):
            text = "Left Hand Down"
        case _ if is_between(up, [130, 0, 0, 0], [180, 30, 30, 30]):
            text = "Right Hand Down"

    if text != 0:
        final.append(text)
        text = 0

    match down:  ## Legs position condition
        case _ if is_between(down, [80, 80, -1, -1], [100, 100, 180, 180]):
            text = "Sitting Down"
        case _ if is_between(down, [160, 160, -1, -1], [180, 180, 180, 180]):
            text = "Standing pose"
        case _ if is_between(down, [80, 160, -1, 160], [100, 180, 180, 180]):
            text = "Standing on left leg"
        case _ if is_between(down, [160, 80, 160, -1], [180, 100, 180, 180]):
            text = "Standing on Right leg"

    if text != 0:
        final.append(text)
        text = 0
    return final

# List of common colors in OpenCV (BGR format)
colors = [
    (0, 0, 255),      # Red
    (0, 255, 0),      # Green
    # (255, 0, 0),      # Blue
    (0, 255, 255),    # Yellow
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (128, 128, 128),  # Gray
    (50, 50, 50),     # Dark Gray
    (200, 200, 200),  # Light Gray
    (0, 0, 128),      # Maroon
    (0, 128, 128),    # Olive
    (128, 0, 128),    # Purple
    (128, 128, 0),    # Teal
    (128, 0, 0),      # Navy
    (0, 165, 255),    # Orange
    (19, 69, 139),    # Brown
    (203, 192, 255),  # Pink
    (230, 216, 173)   # Light Blue
]

"""
Main function starts here to determine the Pose of the subject.
"""

model = yolo("yolov8n-pose.pt", task="pose")  ## defining the model

# cap = cv.VideoCapture("test_video3.mp4")  ## to use a video from the device
cap = cv.VideoCapture(0)  ## To capture from the webcam

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

ret, prev = cap.read()

thresh = 5
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if diff(prev, frame) > thresh:
            print(diff(prev, frame))
            results = model(source=frame)
            frame1 = results[0].plot()
            # thresh += 1
        else:
            frame1 = results[0].plot(
                img=frame
            )  ## to print the annotations on the original image without feeding it into the model.
            print(diff(prev, frame))
            # thresh -= 0.5
        prev = frame
        joints = np.array(results[0].keypoints.xy).astype(
            int
        )  # saving the coordinates in joints variable as int

        img_text = np.zeros((frame1.shape[0], frame1.shape[1] + 200, frame1.shape[2]), dtype=np.uint8)
        img_text[: frame1.shape[0], : frame1.shape[1], : frame1.shape[2]] = frame1
        cv.putText(
            img_text,
            text="Pose:",
            org=(frame1.shape[1], 20),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )
        var = 1
        for j in range(len(joints)): ### This loop ensures to detect pose for more than one persons.
            out = joint_angles(joints[j])
            final = pose(out) ### text variable containg the pose names.
            cv.putText(
            img_text,
            text="Person "+str(j+1)+" :",
            org=(frame1.shape[1], (var+1)*20),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=colors[var],
            thickness=1,
            lineType=cv.LINE_AA,
        )
            var += 1
            for i in range(len(final)):
                cv.putText(
                    img_text,
                    text=final[i],
                    org=(frame1.shape[1], 20 * (var + 1)),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=colors[i+j],
                    thickness=1,
                    lineType=cv.LINE_AA,
                )
                var += 1
        cv.imshow("Frame", img_text)
        # Break the loop on 'q' key press
        if cv.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
