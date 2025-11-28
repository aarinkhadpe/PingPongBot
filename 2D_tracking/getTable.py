# this code is a simple way to get the corners and middle edges of the table for 2D vision (ONE CAMERA)
import ctypes
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mvsdk

# load distortion calibration parameters
with np.load("2D_tracking/dist_calibration.npz") as X:
    cameraMatrix, distCoeffs = [X[i] for i in ('cameraMatrix', 'distCoeffs')]

# detect camera
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 1:
    raise Exception("No camera found")

DevInfo = DevList[0]
hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
print("Opened camera:", DevInfo.acFriendlyName)

# camera settings
mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraPlay(hCamera)
pFrameBuffer = mvsdk.CameraAlignMalloc(1280 * 1024 * 3, 16)

# make this reoslution match the reoslution of the tracking program 
RESIZE_W, RESIZE_H = 640, 480  

# make list for the selected points
click_points = []
point_names = [
    "top-left", "top-mid", "top-right",
    "right-mid", "bottom-right", "bottom-mid",
    "bottom-left", "left-mid"
]

def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(click_points) < len(point_names):
            click_points.append((x, y))
            print(f"{point_names[len(click_points)-1]}: {x}, {y}")
        else:
            print("All points already selected")

window_name = "Click corners/midpoints (Press 'q' to quit, 'r' to reset)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

# main loop
while True:
    try:
        # get frame
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        frame_data = (ctypes.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((FrameHead.iHeight, FrameHead.iWidth, 3))

        # undistort and crop (Keep ROI)
        h, w = frame.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
        x, y, w_roi, h_roi = roi
        cropped = undistorted[y:y+h_roi, x:x+w_roi]

        # flip
        flipped = cv2.flip(cropped, 1)

        # resize
        resized = cv2.resize(flipped, (RESIZE_W, RESIZE_H))

        # show clicked points as red dots
        display = resized.copy()
        for i, (px, py) in enumerate(click_points):
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(display, point_names[i], (px+5, py-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            click_points = []
            print("Points reset.")

    except mvsdk.CameraException as e:
        if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
            print("Camera error:", e)
            break
 
# close the camera feed windows and disconnect from the camera
mvsdk.CameraUnInit(hCamera)
mvsdk.CameraAlignFree(pFrameBuffer)
cv2.destroyAllWindows()

# save the table points
if len(click_points) == len(point_names):
    scaled_points = np.array(click_points, dtype=np.float32)
    np.savez("table_click_points_scaled.npz",
             pixel_points=scaled_points,
             names=point_names,
             width=RESIZE_W,
             height=RESIZE_H)
    print("Succesfully saved scaled and flipped points to table_click_points_scaled.npz")
else:
    print(f"not all points selected :( ({len(click_points)}/{len(point_names)})")
