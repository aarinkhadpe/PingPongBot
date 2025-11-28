import mvsdk
import numpy as np
import cv2
import ctypes

# ------------------ ArUco Setup (OpenCV 4.12 API) ------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ------------------ Initialize MVS Camera ------------------
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 1:
    raise Exception("No camera detected!")

DevInfo = DevList[0]
hCam = mvsdk.CameraInit(DevInfo, -1, -1)
mvsdk.CameraSetIspOutFormat(hCam, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

# Manual exposure
mvsdk.CameraSetAeState(hCam, 0)
mvsdk.CameraSetExposureTime(hCam, 15000)  # adjust as needed
mvsdk.CameraSetAnalogGain(hCam, 40)

mvsdk.CameraPlay(hCam)

cap = mvsdk.CameraGetCapability(hCam)
buf = mvsdk.CameraAlignMalloc(
    cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * 3,
    16
)

print("➡ ArUco detection running (OpenCV 4.12). Press 'q' to quit.")

# ------------------ Main Loop ------------------
try:
    while True:
        # Capture from MVS camera
        pRaw, head = mvsdk.CameraGetImageBuffer(hCam, 200)
        mvsdk.CameraImageProcess(hCam, pRaw, buf, head)
        mvsdk.CameraReleaseImageBuffer(hCam, pRaw)

        frame = np.frombuffer(
            (ctypes.c_ubyte * head.uBytes).from_address(buf),
            dtype=np.uint8
        ).reshape((head.iHeight, head.iWidth, 3))

        # -------- ArUco Detection (OpenCV 4.12 API) --------
        corners, ids, rejected = aruco_detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print("Detected:", ids.flatten())

        # Show image
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    mvsdk.CameraUnInit(hCam)
    mvsdk.CameraAlignFree(buf)
    cv2.destroyAllWindows()
