import cv2, cv
import sys
from datetime import datetime
import ast
import numpy as np
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt
import time
import glob
import math
from sklearn import *
import os
from mpl_toolkits.mplot3d import Axes3D
import tools

# MATCHER ALGORITHMS
CVME_SURF = 1
CVME_SIFT = 2
CVME_ORB = 3

# DEFAULT CONSTANTS
DIST_COEF = np.array([-3.20678032e+01, -6.02849983e-03, -3.21918860e-03, -7.12706263e-02, 2.41369510e-07])
CAM_MATRIX = np.array([[8.84126845e+03, 0.00000000e+00, 3.20129093e+02],
                       [0.00000000e+00, 8.73308727e+03, 2.40511239e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
CAM_WIDTH = 640
CAM_HEIGHT = 480
SMOOTHING_N = 10
DEG_TOLERANCE = 2
NEIGHBORS = 1
SURF_HESSIAN = 1000
SURF_OCTAVES = 4
SURF_LAYERS = 2
SURF_UPRIGHT = 1
SURF_EXTENDED = 1
ORB_FEATURES = 2000
SIFT_FEATURES = 1000
CROSS_CHECK = True
CROP_WIDTH = 400
CROP_HEIGHT = 300
FPS = 25
ZOOM = 0.975

class CVME:
    def __init__(self, cam, features=CVME_SURF):
        self.cam = cam
        self.features = features
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(CAM_MATRIX,
                                                           DIST_COEF,
                                                           None,
                                                           CAM_MATRIX,
                                                           (CAM_WIDTH, CAM_HEIGHT),
                                                           5)
        if features == CVME_SURF:
            self.feature_descriptor = cv2.SURF(SURF_HESSIAN,
                                    nOctaves=SURF_OCTAVES,
                                    nOctaveLayers=SURF_LAYERS,
                                    extended=SURF_EXTENDED,
                                    upright=SURF_UPRIGHT)
        if features == CVME_ORB:
            self.feature_descriptor = cv2.ORB(ORB_FEATURES)
        if features == CVME_SIFT:
            self.feature_descriptor = cv2.SIFT(SIFT_FEATURES)
        self.matcher = cv2.BFMatcher(crossCheck=CROSS_CHECK)    

        self.pts1, self.desc1 = None, None
        self.pts2, self.desc2 = None, None
        self.hist = []
        self.smooth = []
        self.match_hist = []
        self.kp_hist = []
        self.freq_hist = []
    def find_matches(self):
        s = False
        while not s:        
            s, bgr = self.cam.read()
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dst = self.undistort(self.gray)
        self.pts2, self.desc2 = self.pts1, self.desc1
        (self.pts1, self.desc1) = self.feature_descriptor.detectAndCompute(dst, None) # Find key-points
        self.matches = self.matcher.knnMatch(self.desc1, self.desc2, k=NEIGHBORS) # knn-Match descriptor sets
        n = len(self.matches)
        return n
    def calculate_vector(self):
        pairs = []
        for m in self.matches:
            if len(m) != 0:
                #if m.distance < alpha * n.distance: #!TODO support for ratio-test
                pt1 = self.pts1[m[0].queryIdx]
                pt2 = self.pts2[m[0].trainIdx]
                xy1 = (pt1.pt[0], pt1.pt[1])
                xy2 = (pt2.pt[0], pt2.pt[1])
                pairs.append((xy1, xy2))
        projected = [(self.project(pt1), self.project(pt2)) for (pt1, pt2) in pairs]
        vectorized = [self.vectorize(pt1, pt2) for (pt1, pt2) in projected]
        v_all = np.array([v for (v,t) in vectorized])
        t_all = np.array([t for (v,t) in vectorized])
        v_best, t_best = self.hist_filter(v_all, t_all) # Filter for best matches
        t = t_best.mean()
        v = np.median(v_best) #!TODO: estimation for speed, axiom: middle of pack is most likely
        return v, t
    def hist_filter(self, v, t):
        rounded = np.around(t, 0).astype(np.int32)
        bins = [tools.maprange(i, (-180,180), (0,360)) for i in rounded]
        counts = np.bincount(bins)
        mode = np.argmax(counts)
        best = np.isclose(bins, mode, atol=DEG_TOLERANCE) # each bin is equal to --> mode +/- DEG_TOLERANCE
        v = v[best]
        t = t[best]
        return v, t
    def lin_reg(self, x,y):
        x = rtk_smooth
        y = surf_smooth
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y)[0]
        return m, b
    def moving_average(x, n=SMOOTHING_N):
        ret = np.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    def hist_eq(self, gray):
        hist, bins = np.histogram(gray.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        gray_norm = cdf[gray] # Now we have the look-up table
        return gray_norm
    def rmse(self, predictions, targets):
        return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
    def undistort(self, gray):
        return cv2.remap(gray, self.mapx, self.mapy, cv2.INTER_LINEAR) # use linear interpolation
    def project(self, pt, K=ZOOM, w=CAM_WIDTH, h=CAM_HEIGHT):
        """ Project from cam-space to real-space """
        x = pt[0] - w / 2.0
        y = pt[1] - h / 2.0
        X = x * K
        Y = y * K
        return (X,Y)
    def vectorize(self, pt1, pt2, hz=FPS):
        """ Calculate vectors of good matches """    
        (x1, y1) = pt1
        (x2, y2) = pt2
        d = np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        p = np.arctan2((x2 - x1), (y2 - y1))
        t = np.rad2deg(p) # converted to degrees
        v = (3.6 / 1000.0) * (d * float(hz))# convert from mm/s to km/hr
        return (v,t)
    def entropy(self):
        """ Calculate entropy of image """
        hist = cv2.calcHist([self.gray], [0], None, [256], [0,256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 0.00001)
        entropy = -1 * (hist * logs).sum()
        return entropy
    def set_threshold(self, val):
        if self.features == CVME_SURF:
            self.feature_descriptor.hessianThreshold = val
        if self.features == CVME_ORB:
            self.feature_descriptor = cv2.ORB(val)
