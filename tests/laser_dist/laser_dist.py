import numpy as np
import cv2, cv
import sys

# DEFAULT CONSTANTS
CAM_WIDTH = 640
CAM_HEIGHT = 480
DIST_COEF = np.array([-3.20678032e+01, -6.02849983e-03, -3.21918860e-03, -7.12706263e-02, 2.41369510e-07])
CAM_MATRIX = np.array([[8.84126845e+03, 0.00000000e+00, 3.20129093e+02],
                       [0.00000000e+00, 8.73308727e+03, 2.40511239e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
MAPX, MAPY = cv2.initUndistortRectifyMap(CAM_MATRIX,DIST_COEF,None,CAM_MATRIX,(CAM_WIDTH, CAM_HEIGHT),5)

def find_dots_contour(bgr, search_x=(100, 540), search_y=(240, 480)):
    """ """
    dots = []
    hsv = cv2.cvtColor(bgr, cv.CV_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,0,0), (10,255,255))
    mask2 = cv2.inRange(hsv, (240,0,0), (255,255,255))
    mask = mask1 + mask2
    #mask = cv2.remap(mask, MAPX, MAPY, cv2.INTER_LINEAR) # use linear interpolation
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((8,8), np.uint8))
    #mask = cv2.morphologyEx(mask, cv2.MORPH_ELLIPSE, np.ones((3,3), np.uint8))
    #$mask[mask > 0] = 255
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # only proceed if at least one contour was found
    if len(cnts) >= 2:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        cs = sorted(cnts, key=cv2.contourArea)
        for c in cs:
            try:
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                if x > search_x[0] and x < search_x[1] and y > search_y[0] and y < search_y[1]:
                    M = cv2.moments(c)
                    dots.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), radius))
            except:
                pass
    return dots, mask
def find_dots_hough(bgr):
    """
    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghcircles/py_houghcircles.html
    http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    """
    hsv = cv2.cvtColor(bgr, cv.CV_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    dots = []
    try:
        circles = cv2.HoughCircles(gray, cv2.cv.CV_GRADIENT, 1, 60, minRadius=5, maxRadius=15)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            dots.append((i[0],i[1],i[2]))
    except:
        pass
    return dots
def find_dots_sig(bgr):
    dots = []
    hsv = cv2.cvtColor(bgr, cv.CV_BGR2HSV)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    cv2.imshow('', gray)
    cv2.waitKey(0)
    return dots
def calc_dist(pt1, pt2, a=0.007344, b=-2.40728224, c=246.317244):
    x1,y1 = pt1
    x2,y2 = pt2
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    X = a*d**2 + b*d + c 
    return X, d
def maprange(val, a, b):
    (a1, a2), (b1, b2) = a, b
    return  b1 + ((val - a1) * (b2 - b1) / (a2 - a1))

if __name__ == '__main__':
    if len(sys.argv[:]) > 1:
        bgr = cv2.imread(sys.argv[1])
        dots = find_dots_contour(mask)
        pt1 = (dots[1][0], dots[1][1])
        pt2 = (dots[0][0], dots[0][1])
        X, d = calc_dist(pt1, pt2)
        print "Distance (px): %f" % d
        for x,y,r in dots:
            cv2.circle(bgr, (int(x), int(y)), int(r), (255, 0, 0), 2)
        cv2.line(bgr, pt1, pt2, (0,255,0))
        cv2.putText(bgr, str(dist), (320,440), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.imshow('', bgr)
        cv2.waitKey(0)
    else:
        cam = cv2.VideoCapture(1)    
        while True:
            try:
                s, bgr = cam.read()
                if s:
                    dots, M = find_dots_contour(bgr)
                    output = np.dstack((M,M,M))
                    if len(dots) == 2:
                        pt1 = (dots[1][0], dots[1][1])
                        pt2 = (dots[0][0], dots[0][1])
                        X, d = calc_dist(pt1, pt2)
                        print "Distance (px): %f" % d
                        for x,y,r in dots:
                            cv2.circle(output, (int(x), int(y)), int(r), (255, 0, 0), 2)
                        cv2.line(output, pt1, pt2, (0,255,0))
                    else:
                        pass
                    cv2.imshow('', output)
                    if cv2.waitKey(5) == 0:
                        pass
            except KeyboardInterrupt:
                cam.release()
                break
