# This code is meant to track a ping pong ball in 2D (x and y)
# using a single camera

import mvsdk
import numpy as np
import cv2
import ctypes
import matplotlib.pyplot as plt
import time
from collections import deque

# import the calibration data
with np.load("dist_calibration.npz") as X:
    cameraMatrix, distCoeffs = [X[i] for i in ('cameraMatrix', 'distCoeffs')]

# import scaled table points for masking things outside the table (uses corners and middle edges)
data = np.load("table_click_points_scaled.npz")
table_points = data["pixel_points"]  # they are pre-scaled


# connect to the camera using the provided mvsdk
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 1:
    raise Exception("Camera not found")
DevInfo = DevList[0]
hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
print("Opened camera:", DevInfo.acFriendlyName)

# camera settings
mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraSetAeState(hCamera, 0)
mvsdk.CameraSetExposureTime(hCamera, 4000) # change based on brightness - higher exposure means more blur, so make it as low as possible
mvsdk.CameraSetAnalogGain(hCamera, 32)
mvsdk.CameraSetFrameSpeed(hCamera, 1) 
mvsdk.CameraPlay(hCamera)


# assign memory allocation for camera frames:
pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)  #capture one frame to get its dimensions
pFrameBuffer = mvsdk.CameraAlignMalloc(FrameHead.iWidth * FrameHead.iHeight * 3, 16) #assign the proper amount of memory depending on the frame size
mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

# prepare camera undistortion using the camera distortion calibration data from another program
frame_shape = (1024, 1280)
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix, distCoeffs, frame_shape[::-1], 1)

map1, map2 = cv2.initUndistortRectifyMap(
    cameraMatrix, distCoeffs, None, newCameraMatrix,
    frame_shape[::-1], cv2.CV_16SC2)

roi_x, roi_y, roi_w, roi_h = roi

# set the resolution for ALL tracking/processing images - less resolution = faster + less accurate,
# so this should be optimized
RESIZE_W, RESIZE_H = 640, 480

# Mask out regions besides the table using the imported table points from earlier
table_mask = np.zeros((RESIZE_H, RESIZE_W), dtype=np.uint8)
cv2.fillPoly(table_mask, [table_points.astype(np.int32)], 255)


# create tracking variables
ball_positions = []
#prevCircle = None
fps_deque = deque(maxlen=30)
prev_time = time.time()
#tracking_enabled = True

# MAIN LOOP :)
while True:
    try:
        # get a single frame
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        frame_data = (ctypes.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            (FrameHead.iHeight, FrameHead.iWidth, 3))

        # Undistort, crop, resize, flip, mask - the only one that is strictly necessary is undistort
        # the rest are to view the data/frame easier
        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        cropped = undistorted[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        frame_resized = cv2.resize(cropped, (RESIZE_W, RESIZE_H))
        frame_resized = cv2.flip(frame_resized, 1)
        frame_masked = cv2.bitwise_and(frame_resized, frame_resized, mask=table_mask)

        # mask the frame to isolate orange (ping pong ball)
        hsv = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([3, 90, 90]), np.array([30, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # set up openCV contour detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        best_radius = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 2000:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)

            if radius < 4 or radius > 40:
                continue

            if radius > best_radius:
                best_radius = radius
                best_center = (int(x), int(y))

        if best_center is not None:
            x, y = best_center
            prevCircle = np.array([x, y, best_radius])
            ball_positions.append([x, y, 0])

            # draw the detection (green circle and red center point) to confirm it works well - this is also not strictly necessary
            cv2.circle(frame_resized, (x, y), int(best_radius), (0,255,0), 2)
            cv2.circle(frame_resized, (x, y), 2, (0,0,255), -1)

        # FPS display - again not necessary
        now = time.time()
        fps_deque.append(1 / (now - prev_time))
        prev_time = now
        avg_fps = sum(fps_deque) / len(fps_deque)
        cv2.putText(frame_resized, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        #show the regular frame with tracking and the masked frame
        cv2.imshow("Camera View", frame_resized)
        cv2.imshow("Color Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            break

    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
            print("Camera error:", e)
            break



# after p is pressed, close the masked and camera feed windows, and let go of the camera
mvsdk.CameraUnInit(hCamera)
mvsdk.CameraAlignFree(pFrameBuffer)
cv2.destroyAllWindows()

# convert the python list of balls into a numpy array for faster/more organized processing
ball_positions = np.array(ball_positions)

# Split up the array of ball positions into arrays of each shot

shot = []
pointStart = 0
direction = ball_positions[1, 0] > ball_positions[0, 0]  # True = left > right

for i in range(1, len(ball_positions)-1):

    current_direction = ball_positions[i+1, 0] > ball_positions[i, 0]
    if current_direction != direction: # this means: if the direction has changed
        # End of current shot
        shot.append(ball_positions[pointStart:i+1])
        pointStart = i+1
        direction = current_direction

# Add the last shot
if pointStart < len(ball_positions):
    shot.append(ball_positions[pointStart:])


# plot the raw tracking points
plt.figure("Raw Ball Positions")
plt.scatter(ball_positions[:, 0], ball_positions[:, 1], color='red', s=20)
plt.xlim(0, RESIZE_W)
plt.ylim(0, RESIZE_H)
plt.gca().invert_yaxis()
plt.title("Raw Ball Positions")
plt.show(block=False)

# set up the graph for shots
fig, ax = plt.subplots()
ax.set_xlim(0, RESIZE_W)
ax.set_ylim(0, RESIZE_H)
ax.invert_yaxis()  # this is for matching image coordinates
ax.set_title("Ball Trajectory (Segmented by Shots)")

# Draw the table outline in blue
ax.add_patch(plt.Polygon(table_points, closed=True, fill=False, color='blue', linewidth=2))

# Color each shot differently and plot (NOT A REGRESSION even though it might look like it)
colors = plt.cm.tab10(np.linspace(0, 1, len(shot)))

for i, s in enumerate(shot):
    ax.plot(s[:, 0], s[:, 1], '-', color=colors[i], label=f"Shot {i+1}")

ax.legend()
plt.show()