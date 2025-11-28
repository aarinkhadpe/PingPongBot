import mvsdk
import numpy as np
import cv2
import ctypes
import glob
import os
import time

# checkerboard settings
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 26.0  # mm

SAVE_LEFT = "cam_left"
SAVE_RIGHT = "cam_right"
os.makedirs(SAVE_LEFT, exist_ok=True)
os.makedirs(SAVE_RIGHT, exist_ok=True)

print("Press '1' to save image pair")
print("Press 'q' to finish and run stereo calibration\n")

# detect both cameras
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 2:
    raise Exception("Need two cameras connected!")

# left camera initialization and settings
hLeft = mvsdk.CameraInit(DevList[0], -1, -1) # change DevList[0] to 1 if camera is wrong
capLeft = mvsdk.CameraGetCapability(hLeft)
mvsdk.CameraSetIspOutFormat(hLeft, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraPlay(hLeft)
bufsizeLeft = capLeft.sResolutionRange.iWidthMax * capLeft.sResolutionRange.iHeightMax * 3
pBufLeft = mvsdk.CameraAlignMalloc(bufsizeLeft, 16)

# right camera initialization and settings
hRight = mvsdk.CameraInit(DevList[1], -1, -1) # change DevList[1] to 0 if camera is wrong
capRight = mvsdk.CameraGetCapability(hRight)
mvsdk.CameraSetIspOutFormat(hRight, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
mvsdk.CameraPlay(hRight)
bufsizeRight = capRight.sResolutionRange.iWidthMax * capRight.sResolutionRange.iHeightMax * 3
pBufRight = mvsdk.CameraAlignMalloc(bufsizeRight, 16)

# live feed loop and image capturing based on keyboard input
img_id = 0
try:
    while True:
        # get left frame
        pRawL, headL = mvsdk.CameraGetImageBuffer(hLeft, 100)
        mvsdk.CameraImageProcess(hLeft, pRawL, pBufLeft, headL)
        mvsdk.CameraReleaseImageBuffer(hLeft, pRawL)
        frameL = np.frombuffer((ctypes.c_ubyte * headL.uBytes).from_address(pBufLeft),
                               dtype=np.uint8).reshape(headL.iHeight, headL.iWidth, 3)

        # get right frame
        pRawR, headR = mvsdk.CameraGetImageBuffer(hRight, 100)
        mvsdk.CameraImageProcess(hRight, pRawR, pBufRight, headR)
        mvsdk.CameraReleaseImageBuffer(hRight, pRawR)
        frameR = np.frombuffer((ctypes.c_ubyte * headR.uBytes).from_address(pBufRight),
                               dtype=np.uint8).reshape(headR.iHeight, headR.iWidth, 3)

        # display both frames...this creates the live feed 
        cv2.imshow("Left Camera", frameL)
        cv2.imshow("Right Camera", frameR)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('1'):
            left_path = os.path.join(SAVE_LEFT, f"left_{img_id:03d}.jpg")
            right_path = os.path.join(SAVE_RIGHT, f"right_{img_id:03d}.jpg")

            cv2.imwrite(left_path, frameL)
            cv2.imwrite(right_path, frameR)

            print(f"Saved Pair {img_id}: {left_path}, {right_path}")
            img_id += 1

        if key == ord('q'):
            break

finally:
    mvsdk.CameraUnInit(hLeft)
    mvsdk.CameraUnInit(hRight)
    mvsdk.CameraAlignFree(pBufLeft)
    mvsdk.CameraAlignFree(pBufRight)
    cv2.destroyAllWindows()


# make a stereo calibration function using OpenCV

def stereo_calibrate(show_rectified=True):
    print("\nRunning stereo calibration")

    left_files = sorted(glob.glob(os.path.join(SAVE_LEFT, "*.jpg")))
    right_files = sorted(glob.glob(os.path.join(SAVE_RIGHT, "*.jpg")))

    num = min(len(left_files), len(right_files))
    print(f"Found {num} image pairs.\n")

    if num < 5:
        raise Exception("ERROR: Need at least 5 good image pairs")

    # Object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                           0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    for i in range(num):
        imgL = cv2.imread(left_files[i])
        imgR = cv2.imread(right_files[i])

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD)
        retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD)

        if retL and retR:
            objpoints.append(objp)

            cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            imgpointsL.append(cornersL2)
            imgpointsR.append(cornersR2)

            print(f"Pair {i} is good")
        else:
            print(f"ERROR: Pair {i} invalid, skipping pair")

    h, w = grayL.shape

    print("\nRunning left camera calibration")
    _, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, (w,h), None, None)

    print("\nRunning right camera calibration")
    _, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, (w,h), None, None)

    print("\nrunning stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC
    retval, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        K1, D1,
        K2, D2,
        (w, h),
        flags=flags
    )

    print("\nStereo calibration completed")
    np.savez("stereo_calibration.npz",
        K1=K1, D1=D1,
        K2=K2, D2=D2,
        R=R, T=T, E=E, F=F
    )
    print("\nSaved → stereo_calibration.npz")

    # show a rectified image pair to visually make sure the calibratrion went well and the parameters are good
    # this image shouldnt be TOO warped, it shoud be undistorted and maybe a bit altered. 
    if show_rectified:
        print("\nShowing rectified example image...")
        imgL = cv2.imread(left_files[0])
        imgR = cv2.imread(right_files[0])
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, (w,h), R, T, alpha=0)
        left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w,h), cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w,h), cv2.CV_16SC2)

        rectL = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        rectR = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        combined = np.hstack((rectL, rectR))
        cv2.imshow("Rectified Example (Press any key to close)", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# run the calibration
stereo_calibrate()
