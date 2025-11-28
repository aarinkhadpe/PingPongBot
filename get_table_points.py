import cv2
import numpy as np
import mvsdk
import ctypes


NUM_POINTS = 8  # number of table points (corners + middle edges)

# load stereo calibration parameters
calib = np.load("stereo_calibration.npz")
K1, D1 = calib["K1"], calib["D1"]
K2, D2 = calib["K2"], calib["D2"]
R, T = calib["R"], calib["T"]

img_width, img_height = 1280, 1024  # Max resolution of cameras

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    K1, D1, K2, D2, (img_width, img_height), R, T, alpha=0
)

left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (img_width, img_height), cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (img_width, img_height), cv2.CV_16SC2)

# detect camera
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 2:
    raise Exception("Need TWO cameras connected!")

# Left camera
hLeft = mvsdk.CameraInit(DevList[0], -1, -1)
mvsdk.CameraSetIspOutFormat(hLeft, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraSetAeState(hLeft, 0)
mvsdk.CameraPlay(hLeft)
capLeft = mvsdk.CameraGetCapability(hLeft)
bufLeft = mvsdk.CameraAlignMalloc(capLeft.sResolutionRange.iWidthMax * capLeft.sResolutionRange.iHeightMax * 3, 16)

# Right camera
hRight = mvsdk.CameraInit(DevList[1], -1, -1)
mvsdk.CameraSetIspOutFormat(hRight, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraSetAeState(hRight, 0)
mvsdk.CameraPlay(hRight)
capRight = mvsdk.CameraGetCapability(hRight)
bufRight = mvsdk.CameraAlignMalloc(capRight.sResolutionRange.iWidthMax * capRight.sResolutionRange.iHeightMax * 3, 16)

# set up list for left and right table points
left_points = []
right_points = []
click_stage = "left"  
current_window = "Left Camera"

def mouse_callback(event, x, y, flags, param):
    global left_points, right_points, click_stage, current_window
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_stage == "left":
            left_points.append([x, y])
            print(f"Left point {len(left_points)}: {x},{y}")
            if len(left_points) >= NUM_POINTS:
                click_stage = "right"
                print("Switch to right camera feed")
                current_window = "Right Camera"
        elif click_stage == "right":
            right_points.append([x, y])
            print(f"Right point {len(right_points)}: {x},{y}")

cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Left Camera", mouse_callback)
cv2.setMouseCallback("Right Camera", mouse_callback)

print("Click 8 points on the left camera feed")

# main loop for live feed
while True:
    # Grab left frame
    pRawL, headL = mvsdk.CameraGetImageBuffer(hLeft, 200)
    mvsdk.CameraImageProcess(hLeft, pRawL, bufLeft, headL)
    mvsdk.CameraReleaseImageBuffer(hLeft, pRawL)
    frameL = np.frombuffer((ctypes.c_ubyte * headL.uBytes).from_address(bufLeft), dtype=np.uint8)
    frameL = frameL.reshape((headL.iHeight, headL.iWidth, 3))
    rectL = cv2.remap(frameL, left_map1, left_map2, cv2.INTER_LINEAR)

    # Grab right frame
    pRawR, headR = mvsdk.CameraGetImageBuffer(hRight, 200)
    mvsdk.CameraImageProcess(hRight, pRawR, bufRight, headR)
    mvsdk.CameraReleaseImageBuffer(hRight, pRawR)
    frameR = np.frombuffer((ctypes.c_ubyte * headR.uBytes).from_address(bufRight), dtype=np.uint8)
    frameR = frameR.reshape((headR.iHeight, headR.iWidth, 3))
    rectR = cv2.remap(frameR, right_map1, right_map2, cv2.INTER_LINEAR)

    # Draw clicked points indicated by red dots
    for pt in left_points:
        cv2.circle(rectL, tuple(pt), 5, (0, 255, 0), -1)
    for pt in right_points:
        cv2.circle(rectR, tuple(pt), 5, (0, 255, 0), -1)

    cv2.imshow("Left Camera", rectL)
    cv2.imshow("Right Camera", rectR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('p'):
        # Only proceed if we have all points
        if len(left_points) == NUM_POINTS and len(right_points) == NUM_POINTS:
            print("Computing 3D points...")
            left_pts = np.array(left_points, dtype=float).T
            right_pts = np.array(right_points, dtype=float).T
            pts4d = cv2.triangulatePoints(P1, P2, left_pts, right_pts)
            pts4d /= pts4d[3]
            points_3d = pts4d[:3].T
            np.savez("table_points.npz", pixel_points=points_3d)
            print("Saved 3D points to table_points.npz")
            break
        else:
            print("You must click all points on both cameras")

    elif key == 27:  # press ESC to quit without saving
        print("Exiting")
        break

# close windows and disconnect from cameras
mvsdk.CameraUnInit(hLeft)
mvsdk.CameraUnInit(hRight)
mvsdk.CameraAlignFree(bufLeft)
mvsdk.CameraAlignFree(bufRight)
cv2.destroyAllWindows()
