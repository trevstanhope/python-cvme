import numpy as np
import cv2, cv
import sys

def find_dots_contour(bgr, search_x=(200, 500), search_y=(200, 400)):
    """ """
    dots = []
    hsv = cv2.cvtColor(bgr, cv.CV_BGR2HSV)
    mask = cv2.inRange(hsv, (0,32,128), (32,255,255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((8,8), np.uint8))  
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
    return dots
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
    return X
def maprange(val, a, b):
    (a1, a2), (b1, b2) = a, b
    return  b1 + ((val - a1) * (b2 - b1) / (a2 - a1))

if __name__ == '__main__':
    if len(sys.argv[:]) > 1:
        bgr = cv2.imread(sys.argv[1])
        dots = find_dots_contour(mask)
        pt1 = (dots[1][0], dots[1][1])
        pt2 = (dots[0][0], dots[0][1])
        dist = calc_dist(pt1, pt2)
        print "Distance (cm): %f" % calc_dist(pt1, pt2)
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
                    find_dots_sig(bgr)
                    dots = find_dots_contour(bgr)
                    if len(dots) == 2:
                        pt1 = (dots[1][0], dots[1][1])
                        pt2 = (dots[0][0], dots[0][1])
                        print "Distance (cm): %f" % calc_dist(pt1, pt2)
                        for x,y,r in dots:
                            cv2.circle(bgr, (int(x), int(y)), int(r), (255, 0, 0), 2)
                        cv2.line(bgr, pt1, pt2, (0,255,0))
                    else:
                        pass
                    cv2.imshow('', bgr)
                    if cv2.waitKey(5) == 0:
                        pass
            except KeyboardInterrupt:
                cam.release()
                break
