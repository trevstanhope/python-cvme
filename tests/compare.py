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
    avi_files = glob.glob('data/*.avi')
    FEATURES = sys.argv[1] # the feature-descriptor to use for the comparison
    try:
        if 'ORB' == sys.argv[1]: features=cvme.CVME_ORB
        if 'ORB_HAMMING' == sys.argv[1]: features=cvme.CVME_ORB_HAMMING
        if 'BRISK' == sys.argv[1]: features=cvme.CVME_BRISK
        if 'SURF' == sys.argv[1]: features=cvme.CVME_SURF
        if 'SURF2' == sys.argv[1]: features=cvme.CVME_SURF2
        if 'SIFT' == sys.argv[1]: features=cvme.CVME_SIFT
    except:
        print "usage: python -m tests.compare SURF/SURF2/ORB/SIFT/BRISK THRESH1 THRESH2 ..."
        exit(1)

    for threshold in sys.argv[2:]:
        for avi_file in avi_files:
            csv_file = avi_file.split('.')[0] + '.csv'
            trial = avi_file.split('/')[1].split('.')[0]
            print "=============================="
            print avi_file
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
