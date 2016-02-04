"""
Computer-Vision Motion estimation
Estimate motion of camera system
TODO
- Test FLANN
- Test Ratio-test implementation (i.e. for NEIGHBORS=2)
"""

import cv2, cv
import numpy as np
import scipy.cluster.hierarchy as hcluster
import time
import glob
import math
from sklearn import *
import tools

# MATCHER ALGORITHMS
CVME_SURF = 1
CVME_SIFT = 2
CVME_ORB = 3
CVME_FAST = 4
CVME_BRISK = 5
CVME_SURF2 = 6
CVME_ORB_HAMMING = 7

# DEFAULT CONSTANTS
DIST_COEF = np.array([-3.20678032e+01, -6.02849983e-03, -3.21918860e-03, -7.12706263e-02, 2.41369510e-07])
CAM_MATRIX = np.array([[8.84126845e+03, 0.00000000e+00, 3.20129093e+02],
                       [0.00000000e+00, 8.73308727e+03, 2.40511239e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
CAM_WIDTH = 640
CAM_HEIGHT = 480
DEG_TOLERANCE = 2 # the tolerance for finding vectors of similar direction (was 2)
NEIGHBORS = 1
SURF_HESSIAN = 700
SURF_OCTAVES = 4
SURF_LAYERS = 2
SURF_UPRIGHT = 1
SURF_EXTENDED = 1
ORB_FEATURES = 2000
SIFT_FEATURES = 1000
BRISK_THRESHOLD = 50
RATIO_TEST = 0.7
CROP_WIDTH = 400
CROP_HEIGHT = 300
FPS = 25
ZOOM = 0.975

class CVME:
    def __init__(self, cam, features=CVME_SURF, threshold=None, FLANN=False, dist_coef=None, cam_matrix=None):
        
        # Keyword Args
        self.cam = cam
        self.features = features

        # Constants
        if cam_matrix:
            self.CAM_MATRIX = cam_matrix
        else:
            self.CAM_MATRIX = CAM_MATRIX
        if dist_coef:
            self.DIST_COEF = dist_coef
        else:
            self.DIST_COEF = DIST_COEF
        self.CAM_WIDTH = CAM_WIDTH
        self.CAM_HEIGHT = CAM_HEIGHT
        self.DEG_TOLERANCE = DEG_TOLERANCE
        self.NEIGHBORS = NEIGHBORS
        self.SURF_HESSIAN = SURF_HESSIAN
        self.SURF_OCTAVES = SURF_OCTAVES
        self.SURF_LAYERS = SURF_LAYERS
        self.SURF_UPRIGHT = SURF_UPRIGHT
        self.SURF_EXTENDED = SURF_EXTENDED
        self.ORB_FEATURES = ORB_FEATURES
        self.SIFT_FEATURES = SIFT_FEATURES
        self.BRISK_THRESHOLD = BRISK_THRESHOLD
        self.RATIO_TEST = RATIO_TEST
        self.CROP_WIDTH = CROP_WIDTH
        self.CROP_HEIGHT = CROP_HEIGHT
        self.FPS = FPS
        self.ZOOM = ZOOM
        self.FLANN = FLANN
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.CAM_MATRIX,
                                                           self.DIST_COEF,
                                                           None,
                                                           self.CAM_MATRIX,
                                                           (self.CAM_WIDTH, self.CAM_HEIGHT),
                                                           5)
        # Feature-Detector
        if self.features == CVME_SURF:
            norm = cv2.NORM_L2
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=4,
                                    nOctaveLayers=2,
                                    extended=1,
                                    upright=1)
        elif self.features == CVME_SURF2:
            norm = cv2.NORM_L2
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=2,
                                    nOctaveLayers=2,
                                    extended=0,
                                    upright=1)
        elif self.features == CVME_ORB:
            norm = cv2.NORM_L2
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
        elif self.features == CVME_ORB_HAMMING:
            norm = cv2.NORM_HAMMING
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
        elif self.features == CVME_SIFT:
            norm = cv2.NORM_L2
            if threshold:
                self.SIFT_FEATURES = threshold
            self.feature_descriptor = cv2.SIFT(self.SIFT_FEATURES)
        elif self.features == CVME_BRISK:
            norm = cv2.NORM_HAMMING
            if threshold:
                self.BRISK_THRESHOLD = threshold
            self.feature_descriptor = cv2.BRISK(self.BRISK_THRESHOLD)
        else:
            raise Exception("Unrecognized feature-descriptor specified")

        # FLANN Matcher
        if FLANN:
            if norm == cv2.NORM_L2:
                flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            else:
                flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                   table_number = 6, # 12
                                   key_size = 12,     # 20
                                   multi_probe_level = 1) #2
            matcher = cv2.FlannBasedMatcher(flann_params, {}) # bug : need to pass empty dict 
        # Otherwise use the Brute-Force Matcher
        else:
            if self.NEIGHBORS == 1:
                if norm == cv2.NORM_HAMMING:
                    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                else:
                    self.matcher = cv2.BFMatcher(crossCheck=True) # use cross checking if N=1
            elif self.NEIGHBORS == 2:
                self.matcher = cv2.BFMatcher() # use ratio-test if N=2
            else:
                raise Exception("Only 1 or 2 Neighbors are supported for the matcher")
        
        # Empty Variables
        self.pts1, self.desc1 = None, None
        self.pts2, self.desc2 = None, None
    def find_matches(self):
        s = False
        while not s:        
            s, bgr = self.cam.read()
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dst = self.undistort(self.gray) # apply undistortion remap
        self.pts2, self.desc2 = self.pts1, self.desc1 # copy previous keypoints
        (self.pts1, self.desc1) = self.feature_descriptor.detectAndCompute(dst, None) # Find key-points between set1 and set2
        self.matches = self.matcher.knnMatch(self.desc1, self.desc2, k=self.NEIGHBORS) # knn-Match descriptor sets
        m = len(self.matches)
        return m # returns total matches found
    def calculate_vector(self):
        pairs = []
        if self.NEIGHBORS == 1:
            for m in self.matches:
                if len(m) != 0:
                    pt1 = self.pts1[m[0].queryIdx]
                    pt2 = self.pts2[m[0].trainIdx]
                    xy1 = (pt1.pt[0], pt1.pt[1])
                    xy2 = (pt2.pt[0], pt2.pt[1])
                    pairs.append((xy1, xy2))
        elif self.NEIGHBORS == 2:
            for m,n in self.matches:
                if m.distance < self.RATIO_TEST * n.distance:
                    pt1 = self.pts1[m.queryIdx]
                    pt2 = self.pts2[m.trainIdx]
                    xy1 = (pt1.pt[0], pt1.pt[1])
                    xy2 = (pt2.pt[0], pt2.pt[1])
                    pairs.append((xy1, xy2))
        vectorized = [self.vectorize(pt1, pt2) for (pt1, pt2) in pairs]
        v_all = np.array([v for (v,t) in vectorized])
        t_all = np.array([t for (v,t) in vectorized])
        if len(v_all) != 0:
            v_best, t_best = self.hist_filter(v_all, t_all) # Filter for best matches
            t = t_best.mean()
            v = np.median(v_best) #!TODO: estimation for speed, axiom: middle of pack is most likely
            n = len(v_all)
        else:
            t = 'NaN'
            v = 'NaN'
            n = 0
        p = len(pairs)
        return v, t, n, p # returns speed, direction, number of valid matches, and number of vector-pairs
    def hist_filter(self, v, t):
        rounded = np.around(t, 0).astype(np.int32)
        bins = [tools.maprange(i, (-180,180), (0,360)) for i in rounded]
        counts = np.bincount(bins)
        mode = np.argmax(counts)
        best = np.isclose(bins, mode, atol=self.DEG_TOLERANCE)
        v = v[best]
        t = t[best]
        return v, t
    def lin_reg(self, x, y):
        x = rtk_smooth
        y = surf_smooth
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y)[0]
        return m, b
    def moving_average(x, n=5):
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
        """ Apply undistort remap to current image """
        return cv2.remap(gray, self.mapx, self.mapy, cv2.INTER_LINEAR) # use linear interpolation
    def vectorize(self, pt1, pt2):
        """ Calculate vectors of good matches """    
        (x1, y1) = pt1
        (x2, y2) = pt2
        d = self.ZOOM * np.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        p = np.arctan2((x2 - x1), (y2 - y1))
        t = np.rad2deg(p) # converted to degrees
        v = (3.6 / 1000.0) * (d * float(self.FPS))# convert from mm/s to km/hr
        return (v,t)
    def entropy(self):
        """ Calculate entropy of image """
        hist = cv2.calcHist([self.gray], [0], None, [256], [0,256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 0.00001)
        entropy = -1 * (hist * logs).sum()
        return entropy
