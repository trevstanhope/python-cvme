from cvme import cvme
import cv2
import sys
import time

if __name__ == '__main__':
    f = sys.argv[1]
    cam = cv2.VideoCapture(f)
    root = cvme.CVME(cam, features=cvme.CVME_SURF)
    Q = []
    while True:
        try:
            t1 = time.time()
            n = root.find_matches()
            e = root.entropy()
            (v,t) = root.calculate_vector()
            t2 = time.time()
            hz = 1 / (t2 - t1)
            Q.append((n,v,t,e,hz))
            # root.set_threshold()
            print n, v, hz
        except KeyboardInterrupt as e:
            Q = [list(t) for t in zip(*Q)]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Q[4], Q[1], Q[3])
            plt.show()
            break
        except Exception as e:
            print(str(e))
