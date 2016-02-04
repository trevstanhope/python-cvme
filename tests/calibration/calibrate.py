import numpy as np
import cv2
import glob
import sys

cx = 8
cy = 6

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cx*cy, 3), np.float32)
objp[:,:2] = np.mgrid[0:cy, 0:cx].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Iterate through each image
fname = sys.argv[1]
print "Image:\t%s" % fname
img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (cy,cx), None)

# If found, add object points, image points (after refining them)
if ret == True:
    print "Points:\t%d" % len(corners)
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria) # refine corners 
    imgpoints.append(corners)
    for pt in corners:
        cv2.circle(img, tuple(pt[0]), 4, (0,255,0), thickness=1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
else:
    print "Failed to find corners!"
    exit(1)

# Find calibration matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))
print "Distance coefficients", dist
print "Camera matrix", mtx
print "New Camera matrix", newcameramtx

# Manually remap to undistort
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# Undistort as singular function
#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.imwrite('dst.jpg', dst)

# Crop the image
x, y, w, h = roi
crop = dst[y:y+h, x:x+w]
#cv2.imshow('crop', crop)
#cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()
