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
    for threshold in sys.argv[1:]:
        SURF_THRESHOLD = int(threshold)
        ORB_THRESHOLD = int(threshold)
        SIFT_THRESHOLD = int(threshold)
        for avi_file in avi_files:
            csv_file = avi_file.split('.')[0] + '.csv'
            trial = avi_file.split('/')[1].split('.')[0]
            print "=============================="
            print avi_file
            cam1 = cv2.VideoCapture(avi_file)
            cam2 = cv2.VideoCapture(avi_file)
            cam3 = cv2.VideoCapture(avi_file)
            gps_file = open(csv_file, 'r')
            gps = gps_file.readline()
            surf = cvme.CVME(cam2, features=cvme.CVME_SURF, threshold=SURF_THRESHOLD)
            orb = cvme.CVME(cam1, features=cvme.CVME_ORB, threshold=ORB_THRESHOLD)
            sift = cvme.CVME(cam3, features=cvme.CVME_SIFT, threshold=SIFT_THRESHOLD)
            surf_res = []
            orb_res = []
            sift_res = []
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
                        (surf_v, surf_t, surf_m, surf_n, surf_p, surf_hz) = ('NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN')
                        surf_t1 = time.time()
                        surf_m = surf.find_matches()
                        (surf_v, surf_t, surf_n, surf_p) = surf.calculate_vector()
                        surf_t2 = time.time()
                        surf_hz = 1 / (surf_t2 - surf_t1)
                    except Exception as e:
                        print str(e)

                    # SIFT
                    try:
                        (sift_v, sift_t, sift_m, sift_n, sift_p, sift_hz) = ('NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN')
                        sift_t1 = time.time()
                        sift_m = sift.find_matches()
                        (sift_v, sift_t, sift_n, sift_p) = sift.calculate_vector()
                        sift_t2 = time.time()
                        sift_hz = 1 / (sift_t2 - sift_t1)
                    except Exception as e:
                        print str(e)

                    # ORB
                    try:
                        (orb_v, orb_t, orb_m, orb_n, orb_p, orb_hz) = ('NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN')
                        orb_t1 = time.time()
                        orb_m = orb.find_matches()
                        (orb_v, orb_t, orb_n, orb_p) = orb.calculate_vector()
                        orb_t2 = time.time()
                        orb_hz = 1 / (orb_t2 - orb_t1)
                    except Exception as e:
                        print str(e)
                        
                        
                    # Compile results
                    surf_res.append((i, rtk, surf_p, surf_m, surf_n, surf_hz, surf_t, surf_v))
                    orb_res.append((i, rtk, orb_p, orb_m, orb_n, orb_hz, orb_t, orb_v))
                    sift_res.append((i, rtk, sift_p, sift_m, sift_n, sift_hz, sift_t, sift_v))
                    
                    # Show Progress
                    sys.stdout.flush()
                    sys.stdout.write("\r%2.2f\t%2.1f\t%2.1f(%2.1f)\t%2.1f(%2.1f)\t%2.1f(%2.1f)" %
                        (i/25.0,
                         float(rtk),
                         float(surf_v), float(surf_hz),
                         float(orb_v), float(orb_hz),
                         float(sift_v), float(sift_hz)
                         ))
                    sys.stdout.flush()
                except KeyboardInterrupt as e:
                    break
                except Exception as e:
                    print str(e)

            # Write SURF output
            ensure_dir('output/' + 'SURF-' + str(SURF_THRESHOLD) + '/' + trial + '.csv')
            with open('output/' + 'SURF-' + str(SURF_THRESHOLD) + '/' + trial + '.csv', 'w') as outfile:
                h = ['n','rtk','p','m','n','hz','t','v','\n']
                outfile.write(','.join(h))
                for r in surf_res:
                    n = [str(i) for i in r] + ['\n']
                    outfile.write(','.join(n))

            # Write ORB output
            ensure_dir('output/' + 'ORB-' + str(ORB_THRESHOLD) + '/' + trial + '.csv')
            with open('output/' + 'ORB-' + str(ORB_THRESHOLD) + '/' + trial + '.csv', 'w') as outfile:
                h = ['n','rtk','p','m','n','hz','t','v','\n']
                outfile.write(','.join(h))
                for r in orb_res:
                    n = [str(i) for i in r] + ['\n']
                    outfile.write(','.join(n))

            # Write SURF output
            ensure_dir('output/' + 'SIFT-' + str(SIFT_THRESHOLD) + '/' + trial + '.csv')
            with open('output/' + 'SIFT-' + str(SIFT_THRESHOLD) + '/' + trial + '.csv', 'w') as outfile:
                h = ['n','rtk','p','m','n','hz','t','v','\n']
                outfile.write(','.join(h))
                for r in sift_res:
                    n = [str(i) for i in r] + ['\n']
                    outfile.write(','.join(n))
