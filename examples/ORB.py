from cvme import cvme
import cv2
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    f = sys.argv[1]
    cam = cv2.VideoCapture(f + '.avi')
    gps_file = open(f + '.csv', 'r')
    gps =  gps_file.readline()
    root = cvme.CVME(cam, features=cvme.CVME_ORB)
    res = []
    i = 0
    while True:
        try:
            i+=1
            gps =  gps_file.readline()
            gps_parsed = gps.split(',')
            if gps_parsed[0] == '': break
            gps_parsed.pop()
            rtk = float(gps_parsed[-1])
            t1 = time.time()
            n = root.find_matches()
            e = root.entropy()
            (v,t,m,p) = root.calculate_vector()
            t2 = time.time()
            hz = 1 / (t2 - t1)
            res.append((i,n,hz,t,e,v,rtk))
            print("%d\t%2.1f\t%2.2f\t%2.2f") % (n, hz, v, rtk, (rtk - v))
        except KeyboardInterrupt as e:
            break
        except Exception as e:
            print(str(e))
    try:
        vec = [list(t) for t in zip(*res)]
        fig = plt.figure()
        ax1 = fig.add_subplot(211, projection='3d')
        ax1.scatter(vec[4], vec[1], vec[3])
        ax2 = fig.add_subplot(212)        
        ax2.plot(vec[0], vec[-1])
        ax2.plot(vec[0], vec[-2])
        plt.show()
    except Exception as e:
        print str(e)
