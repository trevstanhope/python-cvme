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
import skfuzzy as sf
from scipy.cluster.vq import *

# MATCHER ALGORITHMS
CVME_USURFEx = 1
CVME_SIFT = 2
CVME_ORB = 3
CVME_FAST = 4
CVME_BRISK = 5
CVME_USURFEx_N2 = 6
CVME_ORB_HAMMING = 7
CVME_ORB_HAMMING_N2 = 8
CVME_USURF = 9
CVME_USURF_N2 = 10
CVME_ORB_FAST = 11
CVME_ORB_FAST_N2 = 12
CVME_ORB_HAMMING2 = 13
CVME_ORB_HAMMING2_N2 = 14
CVME_ORB_HAMMINGCL = 15
CVME_ORB_HAMMINGCL_N2 = 16
CVME_SIFT_N2 = 17

# FILTER METHODS
CVME_HIST = 1
CVME_HCLUSTER = 2
CVME_FUZZY = 3

# EQUALIZATION METHODS
CVME_CLAHE = 1
CVME_HISTEQ = 2

# DEFAULT CONSTANTS
DIST_COEF = np.array([-3.20678032e+01, -6.02849983e-03, -3.21918860e-03, -7.12706263e-02, 2.41369510e-07])
CAM_MATRIX = np.array([[8.84126845e+03, 0.00000000e+00, 3.20129093e+02],
                       [0.00000000e+00, 8.73308727e+03, 2.40511239e+02],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
CAM_WIDTH = 640
CAM_HEIGHT = 480
DEG_TOLERANCE = 2 # the tolerance for finding vectors of similar direction (was 2)
NEIGHBORS = 2
SURF_HESSIAN = 700
ORB_FEATURES = 500
SIFT_FEATURES = 1000
BRISK_THRESHOLD = 50
RATIO_TEST = 0.7
CROP_WIDTH = 640
CROP_HEIGHT = 480
FPS = 25
ZOOM = 0.975
CLIP_LIMIT = 2.0
TILE_GRID_SIZE = (8,8)

class CVME:
    def __init__(self,
                 cam,
                 features=CVME_ORB,
                 threshold=None,
                 filt=CVME_HIST,
                 dist_coef=None,
                 cam_matrix=None,
                 equalize=None):
        
        # Keyword Args
        self.cam = cam
        self.features = features
        self.filt = filt
        self.equalize = equalize
        
        # Optional Args
        if cam_matrix:
            self.CAM_MATRIX = cam_matrix
        else:
            self.CAM_MATRIX = CAM_MATRIX
        if dist_coef:
            self.DIST_COEF = dist_coef
        else:
            self.DIST_COEF = DIST_COEF
            
        # Constants
        self.CAM_WIDTH = CAM_WIDTH
        self.CAM_HEIGHT = CAM_HEIGHT
        self.DEG_TOLERANCE = DEG_TOLERANCE
        self.SURF_HESSIAN = SURF_HESSIAN
        self.ORB_FEATURES = ORB_FEATURES
        self.SIFT_FEATURES = SIFT_FEATURES
        self.BRISK_THRESHOLD = BRISK_THRESHOLD
        self.RATIO_TEST = RATIO_TEST
        self.CROP_WIDTH = CROP_WIDTH
        self.CROP_HEIGHT = CROP_HEIGHT
        self.FPS = FPS
        self.ZOOM = ZOOM
        self.CLIP_LIMIT = CLIP_LIMIT
        self.TILE_GRID_SIZE = TILE_GRID_SIZE
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.CAM_MATRIX,
                                                           self.DIST_COEF,
                                                           None,
                                                           self.CAM_MATRIX,
                                                           (self.CAM_WIDTH, self.CAM_HEIGHT),
                                                           5)
        
        ## Feature-Detector
        # U-SURF (Cross-Check)
        if self.features == CVME_USURF:
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=2,
                                    nOctaveLayers=4,
                                    extended=0,
                                    upright=1)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(crossCheck=True)
        # U-SURF (Ratio-Test)    
        elif self.features == CVME_USURF_N2:
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=2,
                                    nOctaveLayers=4,
                                    extended=0,
                                    upright=1)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher()
        # U-SURF-Ex (Cross-Check)
        elif self.features == CVME_USURFEx:
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=2,
                                    nOctaveLayers=4,
                                    extended=1,
                                    upright=1)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(crossCheck=True)
        # U-SURF-Ex (Ratio-Test)
        elif self.features == CVME_USURFEx_N2:
            if threshold:
                self.SURF_HESSIAN = threshold
            self.feature_descriptor = cv2.SURF(self.SURF_HESSIAN,
                                    nOctaves=2,
                                    nOctaveLayers=4,
                                    extended=1,
                                    upright=1)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher()
        # ORB (Cross-Check)
        elif self.features == CVME_ORB:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(crossCheck=True)
        # ORB-HAMMING (Cross-Check)
        elif self.features == CVME_ORB_HAMMING:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # ORB-HAMMING (Ratio-Test)
        elif self.features == CVME_ORB_HAMMING_N2:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # ORB-HAMMING-CLAHE (Cross-Check)
        elif self.features == CVME_ORB_HAMMINGCL:
            self.equalize = CVME_CLAHE
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # ORB-HAMMING-CLAHE (Ratio-Test)
        elif self.features == CVME_ORB_HAMMINGCL_N2:
            self.equalize = CVME_CLAHE
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # ORB-HAMMING2 (Cross-Check)
        elif self.features == CVME_ORB_HAMMING2:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES, WTA_K=4)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        # ORB-HAMMING2 (N2)
        elif self.features == CVME_ORB_HAMMING2_N2:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES, WTA_K=4)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING2)
        # ORB-FAST (Cross-check)
        elif self.features == CVME_ORB_FAST:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES, scoreType=cv2.ORB_FAST_SCORE)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # ORB-FAST (Ratio-Test)
        elif self.features == CVME_ORB_FAST_N2:
            if threshold:
                self.ORB_FEATURES = threshold
            self.feature_descriptor = cv2.ORB(self.ORB_FEATURES, scoreType=cv2.ORB_FAST_SCORE)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        # SIFT (Cross-Check)
        elif self.features == CVME_SIFT:
            if threshold:
                self.SIFT_FEATURES = threshold
            self.feature_descriptor = cv2.SIFT(self.SIFT_FEATURES)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(crossCheck=True)
        # SIFT (Ratio-Test)
        elif self.features == CVME_SIFT_N2:
            if threshold:
                self.SIFT_FEATURES = threshold
            self.feature_descriptor = cv2.SIFT(self.SIFT_FEATURES)
            self.NEIGHBORS = 2
            self.matcher = cv2.BFMatcher()
        # BRISK (Cross-Check)
        elif self.features == CVME_BRISK:
            if threshold:
                self.BRISK_THRESHOLD = threshold
            self.feature_descriptor = cv2.BRISK(self.BRISK_THRESHOLD)
            self.NEIGHBORS = 1
            self.matcher = cv2.BFMatcher(crossCheck=True)
        else:
            raise Exception("Unrecognized feature-descriptor")

        ## Equalization
        if self.equalize == CVME_CLAHE:
            self.clahe = cv2.createCLAHE(clipLimit=self.CLIP_LIMIT, tileGridSize=self.TILE_GRID_SIZE)
            
        ## Empty Variables
        self.pts1, self.desc1 = None, None
        self.pts2, self.desc2 = None, None
    def find_matches(self):
        s = False
        while not s:        
            s, bgr = self.cam.read()
        self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.equalize == CVME_CLAHE:
            self.gray = self.clahe.apply(self.gray)
        elif self.equalize == CVME_HISTEQ:
            self.gray = self.hist_eq(self.gray)
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
        V = [v for (v,t) in vectorized]
        T = [t for (v,t) in vectorized]
        v_all = np.array(V)
        t_all = np.array(T)
        if len(v_all) != 0:
            if self.filt == CVME_HIST:
                v_best, t_best = self.hist_filter(v_all, t_all) # Filter for best matches
            elif self.filt == CVME_HCLUSTER:
                v_best, t_best = self.hcluster_filter(v_all, t_all) # Filter for best matches
            elif self.filt == CVME_FUZZY:
                v_best, t_best = self.fuzzy_filter(v_all, t_all) # Filter for best matches
            else:
                raise Exception("Bad filtering method!")
            t = np.median(t_best)
            v = np.median(v_best) #!TODO: estimation for speed, axiom: middle of pack is most likely
            n = len(v_all)
        else:
            t = 'NaN'
            v = 'NaN'
            n = 0
        p = len(pairs)
        return v, t, n, p # returns speed, direction, number of valid matches, and number of vector-pairs
    def hist_filter(self, v, t):
        t_rounded = np.around(t, 0).astype(np.int32)
        v_rounded = np.around(v, 1).astype(np.int32)
        t_bins = [tools.maprange(i, (-180,180), (0,360)) for i in t_rounded]
        v_bins = [i for i in v_rounded]
        t_counts = np.bincount(t_bins)
        t_mode = np.argmax(t_counts)
        best = np.isclose(t_bins, t_mode, atol=self.DEG_TOLERANCE)
        ## V_r = np.array(np.round(np.array(v[best])*10), np.uint8)
        ## T_r = np.array(np.round(t[best]), np.uint8) 
        ## output = np.zeros((200, 360, 3), np.uint8)
        ## output[:, T_r, 2] = 255
        ## output[V_r[V_r < 200], :, 1] = 255
        ## cv2.imshow('',output)
        ## if cv2.waitKey(5) == 0:
        ##     pass
        v = v[best]
        t = t[best]
        return v, t
    def hcluster_filter(self, v, t):
        X = np.array(zip(v,t))
        T = hcluster.fclusterdata(X, 0.1)
        lens = {}
        best = []
        w, h = X.shape
        for idx, clno in enumerate(T):
            lens.setdefault(clno, 0)
            lens[clno] += 1
        a = list(lens.values())
        b = list(lens.keys())
        clmax = b[a.index(max(a))]
        best = [idx for idx, clno in enumerate(T) if clno == clmax]
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
