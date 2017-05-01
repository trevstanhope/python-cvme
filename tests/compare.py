"""
Iteratively run a given keypoint matcher on the dataset
Compares against RTK GNSS data
Produces .csv files in output named: $FEATURES-$THRESHOLD/$FILENAME.csv
"""

from cvme import cvme
import cv2
import glob
import time
import matplotlib.pyplot as plt
import sys, os
import json

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
        
if __name__ == '__main__':
    avi_files = glob.glob('data/speed/*.avi')
    FEATURES = sys.argv[1] # the feature-descriptor to use for the comparison
    try:
        if 'ORB_FAST' == sys.argv[1]: features=cvme.CVME_ORB_FAST
        if 'ORB_FAST_N2' == sys.argv[1]: features=cvme.CVME_ORB_FAST_N2
        if 'ORB_HAMMING' == sys.argv[1]: features=cvme.CVME_ORB_HAMMING
        if 'ORB_HAMMING_N2' == sys.argv[1]: features=cvme.CVME_ORB_HAMMING_N2
        if 'ORB_HAMMINGCL' == sys.argv[1]: features=cvme.CVME_ORB_HAMMINGCL
        if 'ORB_HAMMINGCL_N2' == sys.argv[1]: features=cvme.CVME_ORB_HAMMINGCL_N2
        if 'ORB_HAMMINGEQ' == sys.argv[1]: features=cvme.CVME_ORB_HAMMINGEQ
        if 'ORB_HAMMINGEQ_N2' == sys.argv[1]: features=cvme.CVME_ORB_HAMMINGEQ_N2
        if 'ORB_HAMMING2' == sys.argv[1]: features=cvme.CVME_ORB_HAMMING2
        if 'ORB_HAMMING2_N2' == sys.argv[1]: features=cvme.CVME_ORB_HAMMING2_N2

        # SURF
        if 'USURFEx' == sys.argv[1]: features=cvme.CVME_USURFEx
        if 'USURFEx_N2' == sys.argv[1]: features=cvme.CVME_USURFEx_N2
        if 'USURF' == sys.argv[1]: features=cvme.CVME_USURF
        if 'USURF_N2' == sys.argv[1]: features=cvme.CVME_USURF_N2
        if 'SURFEx' == sys.argv[1]: features=cvme.CVME_SURFEx
        if 'SURFEx_N2' == sys.argv[1]: features=cvme.CVME_SURFEx_N2
        if 'SURF' == sys.argv[1]: features=cvme.CVME_SURF
        if 'SURF_N2' == sys.argv[1]: features=cvme.CVME_SURF_N2

        # SIFT
        if 'SIFT' == sys.argv[1]: features=cvme.CVME_SIFT
        if 'SIFT_N2' == sys.argv[1]: features=cvme.CVME_SIFT_N2

        # Misc
        if 'BRISK' == sys.argv[1]: features=cvme.CVME_BRISK

    except:
        print "usage: python -m tests.compare ALG THRESH1 THRESH2 ..."
        exit(1)

    for threshold in sys.argv[2:]:
        for avi_file in avi_files:
            csv_file = avi_file.split('.')[0] + '.csv'
            trial = avi_file.split('/')[-1].split('.')[0]
            print "=============================="
            print avi_file, csv_file, trial, threshold 
            cam = cv2.VideoCapture(avi_file)
            gps_file = open(csv_file, 'r')
            gps = gps_file.readline()
            matcher = cvme.CVME(cam, features=features, threshold=int(threshold))
            res = []
            i = 0
            while True:
                try:
                    i+=1
                    gps =  gps_file.readline()
                    gps_parsed = gps.split(',')
                    if gps_parsed[0] == '':
                        print 'parse failed'
                        break
                    gps_parsed.pop()
                    rtk = float(gps_parsed[-1])

                    # SURF
                    try:
                        (v, t, m, n, p, hz) = ('NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN')
                        t1 = time.time()
                        m = matcher.find_matches()
                        (v, t, n, p) = matcher.calculate_vector()
                        t2 = time.time()
                        hz = 1 / (t2 - t1)
                    except Exception as e:
                        print str(e)

                    # Compile results
                    res.append((i, rtk, p, m, n, hz, t, v))

                    # Show Progress
                    sys.stdout.flush()
                    sys.stdout.write("\r%2.2f\t%2.1f\t%2.1f(%2.1f)" % (i/25.0, float(rtk), float(v), float(hz)))
                    sys.stdout.flush()
                except KeyboardInterrupt as e:
                    break
                except Exception as e:
                    print str(e)

            # Write output
            ensure_dir('output/' + FEATURES + '-' + str(threshold) + '/' + trial + '.csv')
            with open('output/' + FEATURES + '-' + str(threshold) + '/' + trial + '.csv', 'w') as outfile:
                h = ['n','rtk','p','m','n','hz','t','v','\n']
                outfile.write(','.join(h))
                for r in res:
                    n = [str(i) for i in r] + ['\n']
                    outfile.write(','.join(n))
