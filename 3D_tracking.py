# This code is meant to plot the 3D coordinates of a ping pong ball using 2 Cameras 
import mvsdk
import numpy as np
import cv2
import ctypes
import time
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# contour tracking settings
RESIZE_W, RESIZE_H = 640, 480
MIN_AREA = 50
MAX_AREA = 2000
MIN_RADIUS = 4
MAX_RADIUS = 40
lower_orange = np.array([5, 120, 120])
upper_orange = np.array([20, 255, 255])
kernel = np.ones((3, 3), np.uint8)

# load the stereo calibration parameters from another program
calib = np.load("stereo_calibration.npz")
K1, D1 = calib["K1"], calib["D1"]
K2, D2 = calib["K2"], calib["D2"]
R, T = calib["R"], calib["T"]

# do the stereo rectification
img_width, img_height = 1280, 1024  # this should be adjusted based on your cameras resolution

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2,
    (img_width, img_height),
    R, T,
    alpha=0
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (img_width, img_height), cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (img_width, img_height), cv2.CV_16SC2)

# detect both cameras
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 2:
    raise Exception("Need TWO cameras connected!")

# below are the settings for the left and right cameras

# left camera settings
hLeft = mvsdk.CameraInit(DevList[0], -1, -1) # chnage DevList[0] to 1 if the camera is the wrong one
mvsdk.CameraSetIspOutFormat(hLeft, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraSetAeState(hLeft, 0)
mvsdk.CameraSetExposureTime(hLeft, 4000) # should be as low as possible
mvsdk.CameraSetAnalogGain(hLeft, 32)
mvsdk.CameraPlay(hLeft)
capLeft = mvsdk.CameraGetCapability(hLeft)
bufLeft = mvsdk.CameraAlignMalloc(capLeft.sResolutionRange.iWidthMax * capLeft.sResolutionRange.iHeightMax * 3, 16)

# right camera settings
hRight = mvsdk.CameraInit(DevList[1], -1, -1) # change DevList[1] to 0 if this camera is the wrong one
mvsdk.CameraSetIspOutFormat(hRight, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraSetAeState(hRight, 0)
mvsdk.CameraSetExposureTime(hRight, 4000) # should be as low as possible
mvsdk.CameraSetAnalogGain(hRight, 32)
mvsdk.CameraPlay(hRight)
capRight = mvsdk.CameraGetCapability(hRight)
bufRight = mvsdk.CameraAlignMalloc(capRight.sResolutionRange.iWidthMax * capRight.sResolutionRange.iHeightMax * 3, 16)

# this function runs a contour detection for an inputted frame. It outputs a center coordinate and
# estimated radius(pixels), the radius is not too useful right now, since the triangulation works 
# by comparing the center coordinates for each image

def find_ball_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_center, best_radius = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue
        if radius > best_radius:
            best_radius = radius
            best_center = (int(x), int(y))
    return best_center, best_radius

print("Stereo tracking has started. Press 'q' to stop tracking and open up a graph of the shots.")

ball_positions = []

try:
    while True:
        # get the left camera frame
        pRawL, headL = mvsdk.CameraGetImageBuffer(hLeft, 200)
        mvsdk.CameraImageProcess(hLeft, pRawL, bufLeft, headL)
        mvsdk.CameraReleaseImageBuffer(hLeft, pRawL)
        frameL = np.frombuffer((ctypes.c_ubyte * headL.uBytes).from_address(bufLeft), dtype=np.uint8)
        frameL = frameL.reshape((headL.iHeight, headL.iWidth, 3))
        rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)

        # get the right camera frame
        pRawR, headR = mvsdk.CameraGetImageBuffer(hRight, 200)
        mvsdk.CameraImageProcess(hRight, pRawR, bufRight, headR)
        mvsdk.CameraReleaseImageBuffer(hRight, pRawR)
        frameR = np.frombuffer((ctypes.c_ubyte * headR.uBytes).from_address(bufRight), dtype=np.uint8)
        frameR = frameR.reshape((headR.iHeight, headR.iWidth, 3))
        rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

        # mask both frames to only include a range of color (orange-ish colors)
        hsvL = cv2.cvtColor(rectL, cv2.COLOR_BGR2HSV)
        maskL = cv2.inRange(hsvL, lower_orange, upper_orange)
        maskL = cv2.morphologyEx(maskL, cv2.MORPH_OPEN, kernel)
        maskL = cv2.morphologyEx(maskL, cv2.MORPH_CLOSE, kernel)

        hsvR = cv2.cvtColor(rectR, cv2.COLOR_BGR2HSV)
        maskR = cv2.inRange(hsvR, lower_orange, upper_orange)
        maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)
        maskR = cv2.morphologyEx(maskR, cv2.MORPH_CLOSE, kernel)

        # pass the masked frames into ball finding function from above, and store the center points for each ball
        cL, rL = find_ball_center(maskL)
        cR, rR = find_ball_center(maskR)

        if cL and cR:
            xl, yl = cL
            xr, yr = cR

            # triangulate the 3D points
            pts4d = cv2.triangulatePoints(P1, P2,
                                          np.array([[xl],[yl]], dtype=float),
                                          np.array([[xr],[yr]], dtype=float))
            pts4d /= pts4d[3]
            X, Y, Z = pts4d[0][0], pts4d[1][0], pts4d[2][0]
            ball_positions.append([X, Y, Z])

        # display the rectified frame. Both frames are also undistorted.
        # This will slow down the fps, and it is optional - mainly for checking that things are good
        cv2.imshow("Left Rectified", rectL)
        cv2.imshow("Right Rectified", rectR)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    mvsdk.CameraUnInit(hLeft)
    mvsdk.CameraUnInit(hRight)
    mvsdk.CameraAlignFree(bufLeft)
    mvsdk.CameraAlignFree(bufRight)
    cv2.destroyAllWindows()

# Load table points
table_data = np.load("table_points.npz")
table_points = table_data['pixel_points']  # This should be shape (8, 3)

# break apart the raw ball coordinates into groups of SHOT coorinates
# the way this works is by checking if the next x-value of the ball shows a different trend than
# the previous ones, then it is a new shot. LOGIC: if the current x-value is greater than the previous, 
# but ther next value is less, than it is a new shot. This works the other way around as well
ball_positions = np.array(ball_positions)
shot = []
if len(ball_positions) > 1:
    pointStart = 0
    direction = ball_positions[1, 0] > ball_positions[0, 0]  # True = left > right

    for i in range(1, len(ball_positions)-1):
        current_direction = ball_positions[i+1, 0] > ball_positions[i, 0]
        if current_direction != direction:
            shot.append(ball_positions[pointStart:i+1])
            pointStart = i+1
            direction = current_direction
    if pointStart < len(ball_positions):
        shot.append(ball_positions[pointStart:])

# plot the 3D points and the table
fig = plt.figure("3D Ball Trajectory by Shot")
ax = fig.add_subplot(111, projection='3d')

# ---- Fill the table polygon ----
table_poly = Poly3DCollection([table_points], facecolor='lightblue', alpha=0.5, edgecolor='k')
ax.add_collection3d(table_poly)

# ---- Plot ball shots ----
colors = plt.cm.tab10(np.linspace(0, 1, len(shot)))
for i, s in enumerate(shot):
    ax.plot(s[:,0], s[:,1], s[:,2], color=colors[i], marker='o', markersize=3, label=f"Shot {i+1}")

# ---- Axes settings ----
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_xlim(-800,800)
ax.set_ylim(-800,800)
ax.set_zlim(0,1500)
ax.invert_zaxis()
ax.legend()
plt.show()
