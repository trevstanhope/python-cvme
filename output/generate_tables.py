import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as sig
import scipy as sp
import time
import scipy.stats as stats
from pykalman import KalmanFilter
import matplotlib
import sklearn as skl

#font = {
#    "family" : 'normal',
#    "size" : 14
#}
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14

def lin_fit(x,y):
    fitfunc = lambda m, x: m * x
    errfunc = lambda p,x,y: fitfunc(p,x) - y
    a_init = 1.0
    a, s = sp.optimize.leastsq(errfunc, np.array(a_init), args=(x,y))
    p = np.poly1d([a,0])
    yhat = p(x)
    ybar = np.sum(y) / float(len(y))
    ssreg = np.sum((yhat - ybar)**2)
    sstot = np.sum((y - ybar)**2)
    rsquare = ssreg / sstot
    ttest, pval = sp.stats.ttest_1samp(np.abs(x - y), 0.0)
    return a[0], rsquare, ttest, pval
def poly2d(df, kx='srtk', ky='se'):
    x = np.array(df[kx][~np.isnan(df[ky])])
    y = np.abs(np.array(df[ky][~np.isnan(df[ky])]))
    coef = np.polyfit(x,y,2)
    X = np.linspace(1,5)
    Y = np.polyval(coef, X)
    return X, Y
def calc_dist(pt1, pt2, a=0.007344, b=-2.40728224, c=246.317244):
    x1,y1 = pt1
    x2,y2 = pt2
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    X = a*d**2 + b*d + c 
    return X
def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
def moving_median(interval, window_size):
    return sig.medfilt(interval, kernel_size=window_size)
def rmse(predictions, targets):
    return np.nanmean(abs(np.array(predictions) - np.array(targets)))
    return np.sqrt(np.nanmean(((np.array(predictions) - np.array(targets)) ** 2.0)))
def p95(predictions, targets):
    if len(predictions) == 0 or len(targets) == 0:
        return np.NAN
    else:
        abs_err = np.abs(np.array(predictions[~np.isnan(targets)]) - np.array(targets[~np.isnan(targets)]))
        return np.percentile(abs_err, 95)
def p95_by_group(y, x, bins, lims):
    offset = (lims[1] - lims[0]) / float(bins)
    groups_a = np.linspace(lims[0], lims[1] - offset, num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1], num=bins)
    groups = zip(groups_a, groups_b)
    P95 = []
    N = []
    for (a,b) in groups:
        N.append(np.sum(np.logical_and(y>a, y<=b)))
        Y = y[np.logical_and(y>a, y<=b)]
        X = x[np.logical_and(y>a, y<=b)]
        v = p95(Y, X)
        if np.isnan(v): print "NAN WARNING in RMSE by Group!"
        P95.append(v) 
    return P95, (groups_a + groups_b) / 2.0
def rmse_by_group(y, x, bins, lims):
    offset = (lims[1] - lims[0]) / bins
    groups_a = np.linspace(lims[0], lims[1] - offset, num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1], num=bins)
    groups = zip(groups_a, groups_b)
    RMSE = []
    N = []
    for (a,b) in groups:
        N.append(np.sum(np.logical_and(y>a, y<=b)))
        Y = y[np.logical_and(y>a, y<=b)]
        X = x[np.logical_and(y>a, y<=b)]
        v = rmse(Y, X)
        if np.isnan(v): print "NAN WARNING in RMSE by Group!"
        RMSE.append(v)
    return RMSE, (groups_a + groups_b) / 2.0
def histogram_by_lims(y, x, lims, bins, minmax):
    a = lims[0]
    b = lims[1]
    return np.histogram(y[np.logical_and(y>a, y<b)] - x[np.logical_and(y>a, y<b)], bins, minmax)    
def normalize(x):
    norm=np.linalg.norm(x)
    if norm==0: 
       return x
    return x/norm

# Iterate through all trials
if __name__ == '__main__':
    SPEED_RANGE = [1,5]
    NUM_GROUPS =  4
    SUB_GROUPS = 1
    SMOOTHING_WINDOW = 5
    ORB_TREATMENTS = ['500','750','250']
    SURF_TREATMENTS = ['2000', '1500', '1000']
    SIFT_TREATMENTS = ['500', '750', '250']
    SURFACES = ['asphault', 'grass', 'gravel', 'residue', 'corn', 'hay']
    ALGORITHMS = ['SURFEx',
                  'SURFEx_N2',
                  'USURFEx',
                  'USURFEx_N2',
                  'ORB_HAMMING',
                  'ORB_HAMMING_N2',
                  'ORB_HAMMINGCL',
                  'ORB_HAMMINGCL_N2',
                  'SIFT',
                  'SIFT_N2'
                  ]
    DEZOOM = 1.01
    LINE_TYPES = {
        '125' : 'dotted',
        '250' : 'dotted',
        '500' : 'dashdot',
        '750' : 'solid',
        '1000' : 'dotted',
        '1500' : 'dashdot',
        '2000' : 'solid',
        '3000' : 'solid'
    }
    CORR_FACTORS = {
        'asphault': 1.00,
        'gravel': 1.00,
        'residue': 0.99,
        'grass': 0.95,
        'hay': 0.93,
        'corn' : 0.96,
        'soy' : 0.65
    }
    TRIAL_BLACKLIST = ['gravel-1', 'gravel-7',
                       'corn-1', 'corn-3', 'corn-4', 'corn-7', 'corn-8', 'corn-11',
                       'hay-1']
    HEADERS = ['rtk','v', 't', 'm', 'p', 'n'] # Column headers (must match the .csv headers!)
    TREATMENTS = [tuple(d.split('-')) for d in os.listdir(".") if os.path.isdir(d)]
    TRIALS = [f.split('/')[-1].split('.')[0] for f in glob.glob("../data/*.csv")]
    HATCHES = {
        "USURF" : "",
        "USURF_N2" : "",
        "SURF" : "",
        "SURF_N2" : "",
        "USURFEx" : "",
        "USURFEx_N2" : "",
        "ORB_HAMMING" : "",
        "ORB_HAMMING_N2" : "",
        "ORB_HAMMING2" : "",
        "ORB_HAMMING2_N2" : "",
        "ORB_HAMMINGCL" : "",
        "ORB_HAMMINGCL_N2" : "",
        "ORB_HAMMINGEQ" : "",
        "ORB_HAMMINGEQ_N2" : "",
        "ORB_L2" : "",
        "ORB_L2_N2" : "",
        "ORB_FAST" : "",
        "ORB_FAST_N2" : "",
        "SIFT" : "",
        "SIFT_N2" : ""        
    }
    #COLORS = {
    #    "USURF" : "darkred",
    #    "USURF_N2" : "red",
    #    "SURF" : "darkred",
    #    "SURF_N2" : "red",
    #    "SURFEx" : "darkred",
    #    "SURFEx_N2" : "red",
    #    "USURFEx" : "darkorange",
    #    "USURFEx_N2" : "yellow",
    #    "ORB_HAMMING" : "green",
    #    "ORB_HAMMING_N2" : "lime",
    #    "ORB_HAMMING2" : "blue",
    #    "ORB_HAMMING2_N2" : "royalblue",
    #    "ORB_HAMMINGCL" : "blue",
    #    "ORB_HAMMINGCL_N2" : "royalblue",
    #    "ORB_HAMMINGEQ" : "blue",
    #    "ORB_HAMMINGEQ_N2" : "royalblue",
    #    "ORB_L2" : "orange",
    #    "ORB_L2_N2" : "darkorange",
    #    "ORB_FAST" : "cyan",
    #    "ORB_FAST_N2" : "darkcyan",
    #    "SIFT" : "purple",
    #    "SIFT_N2" : "magenta",
    #    "RTK" : "red"
    #}
    COLORS = {
        "USURF" : "0.05",
        "USURF_N2" : "0.50",
        "SURF" : "0.05",
        "SURF_N2" : "0.50",
        "SURFEx" : "0.05",
        "SURFEx_N2" : "0.50",
        "USURFEx" : "0.05",
        "USURFEx_N2" : "0.50",
        "ORB_HAMMING" : "0.05",
        "ORB_HAMMING_N2" : "0.50",
        "ORB_HAMMING2" : "0.05",
        "ORB_HAMMING2_N2" : "0.50",
        "ORB_HAMMINGCL" : "0.05",
        "ORB_HAMMINGCL_N2" : "0.50",
        "ORB_HAMMINGEQ" : "0.05",
        "ORB_HAMMINGEQ_N2" : "0.50",
        "ORB_L2" : "0.05",
        "ORB_L2_N2" : "0.50",
        "ORB_FAST" : "0.05",
        "ORB_FAST_N2" : "0.50",
        "SIFT" : "0.05",
        "SIFT_N2" : "0.50",
        "RTK" : "0.05"
    }
    MARKER_SIZE = 60
    MARKERS = {
        "USURF" : ">",
        "USURF_N2" : "<",
        "SURF" : ">",
        "SURF_N2" : "<",
        "SURFEx" : ">",
        "SURFEx_N2" : "<", 
        "USURFEx" : "^",
        "USURFEx_N2" : "v",
        "ORB_HAMMING" : "s",
        "ORB_HAMMING_N2" : "D",
        "ORB_HAMMING2" : "d",
        "ORB_HAMMING2_N2" : "*",
        "ORB_HAMMINGCL" : "o",
        "ORB_HAMMINGCL_N2" : "x",
        "ORB_HAMMINGEQ" : "o",
        "ORB_HAMMINGEQ_N2" : "x",
        "ORB_L2" : "o",
        "ORB_L2_N2" : "x",
        "ORB_FAST" : "o",
        "ORB_FAST_N2" : "x",
        "SIFT" : "p",
        "SIFT_N2" : "H",
        "RTK" : "+"
    }
    SURFACE_LABELS = {
        'asphault' : 'Asphalt',
        'gravel' : 'Gravel',
        'residue' : 'Seedlings',
        'grass' : 'Turf Grass',
        'corn' : 'Corn Residue',
        'hay' : 'Pasture'
    }
    ALGORITHM_LABELS = {
        "USURF" : "SURF (cross-check)",
        "USURF_N2" : "SURF (ratio-test)",
        "SURF" : "SURF (cross-check)",
        "SURF_N2" : "SURF (ratio-test)",
        "SURFEx" : "SURF (cross-check)",
        "SURFEx_N2" : "SURF (ratio-test)",
        "USURFEx" : "U-SURF (cross-check)",
        "USURFEx_N2" : "U-SURF (ratio-test)",
        "ORB_HAMMING" : "ORB (cross-check)",
        "ORB_HAMMING_N2" : "ORB (ratio-test)",
        "ORB_HAMMING2" : "CLORB (cross-check)",
        "ORB_HAMMING2_N2" : "CLORB (ratio-test)",
        "ORB_HAMMINGCL" : "CLORB (cross-check)",
        "ORB_HAMMINGCL_N2" : "CLORB (ratio-test)",
        "ORB_HAMMINGEQ" : "EORB (cross-check)",
        "ORB_HAMMINGEQ_N2" : "EORB (ratio-test)",
        "ORB_L2" : "orange",
        "ORB_L2_N2" : "darkorange",
        "ORB_FAST" : "cyan",
        "ORB_FAST_N2" : "darkcyan",
        "SIFT" : "SIFT (cross-check)",
        "SIFT_N2" : "SIFT (ratio-test)"
    }
    LEGEND = [mpatches.Patch(color=COLORS[alg], label=ALGORITHM_LABELS[alg]) for alg in ALGORITHMS]
    # Make dictionaries to load csv-files into DataFrames by algorithm
    # d_usurfex = {
    #   1000 : {
    #     asphault-1 : <df>
    #     ...
    #   }
    # }
    d_usurfex = { thresh:{} for alg,thresh in TREATMENTS}
    d_usurfex_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_usurf = { thresh:{} for alg,thresh in TREATMENTS}
    d_usurf_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_hamming = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_hamming_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_hamming2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_hamming2_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_fast = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_fast_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_sift = { thresh:{} for alg,thresh in TREATMENTS}
    d_sift_n2 = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb_l2 = { thresh:{} for alg,thresh in TREATMENTS}
    # Make dictionaries to sort DataFrames by surface
    # d_asphault = {
    #   SURF : [<df>, ... ],
    #   ...
    # }    
    d_asphault = {alg: [] for alg in ALGORITHMS}
    d_gravel = {alg: [] for alg in ALGORITHMS}
    d_grass = {alg: [] for alg in ALGORITHMS}
    d_residue = {alg: [] for alg in ALGORITHMS}
    d_corn = {alg: [] for alg in ALGORITHMS}
    d_hay = {alg: [] for alg in ALGORITHMS}

    # Kalman Filter
    kf = KalmanFilter(transition_matrices=[[1,1],[0,1]], transition_covariance=0.01 * np.eye(2))
    print("===============================================================")
    output = open("summary.csv", "w")
    output.write("alg,thresh,surf,t_num,hz,pts,rmse,rmse_raw,rmse_1,rmse_2,rmse_3,rmse_4,p95,p95_raw,p95_1,p95_2,p95_3,p95_4,nans,slope,r_value,p_value,std_err\n")
    for alg,thresh in TREATMENTS:
        for f in glob.glob(os.path.join(alg + '-' + thresh,'*.csv')):
            trial = f.split('/')[-1].split('.')[0]
            surface = trial.split('-')[0]
            trial_num = trial.split('-')[1]
            if (surface in SURFACES) and (alg in ALGORITHMS) and not (trial in TRIAL_BLACKLIST):
                print alg, thresh, surface, trial_num
                df = pd.DataFrame.from_csv(f)
                rtk = df['rtk'] / 3.6
                hz = df['hz']
                v = df['v'] * DEZOOM / 3.6
                nans = np.count_nonzero(np.isnan(v))
                cv = v * CORR_FACTORS[surface]
                mask = np.isnan(cv)
                #cv[mask] = 0 # make nans zero
                cv[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cv[~mask]) # linterp nans
                shz = sig.medfilt(hz, kernel_size=SMOOTHING_WINDOW)
                srtk = sig.savgol_filter(rtk, SMOOTHING_WINDOW, 1)
                sv = sig.medfilt(cv, kernel_size=SMOOTHING_WINDOW)
                se = sv - srtk
                d = 100 * ((sig.medfilt(cv, kernel_size=SMOOTHING_WINDOW) / srtk) - 1)
                dv = np.gradient(sv)
                drtk = np.gradient(srtk) * 25.0
                h, b = histogram_by_lims(v[~np.isnan(v)], rtk[~np.isnan(v)], SPEED_RANGE, 100, [-1,1])
                sh, sb = histogram_by_lims(sv, srtk, SPEED_RANGE, 100, [-1,1])
                RMSE_g, groups = rmse_by_group(sv, srtk, NUM_GROUPS, SPEED_RANGE)
                RMSE = rmse(sv, srtk)
                RMSE_raw = rmse(v, rtk)
                hz_mean = np.mean(hz)
                points = len(hz)
                P95_g, groups = p95_by_group(sv, srtk, NUM_GROUPS, SPEED_RANGE)
                P95 = p95(sv, srtk)
                P95_raw = p95(v[~np.isnan(v)], rtk[~np.isnan(v)])
                #slope, intercept, r_value, p_value, std_err = stats.linregress(srtk,sv)
                slope, r_value, t_test, p_value = lin_fit(srtk,sv)
                df = df.join(pd.DataFrame({'d':d}))
                df = df.join(pd.DataFrame({'h':h}))
                df = df.join(pd.DataFrame({'b':b[:-1]}))
                df = df.join(pd.DataFrame({'sh':sh}))
                df = df.join(pd.DataFrame({'sb':sb[:-1]}))
                df = df.join(pd.DataFrame({'cv':cv}))
                df = df.join(pd.DataFrame({'dv':dv}))
                df = df.join(pd.DataFrame({'drtk':drtk}))
                df = df.join(pd.DataFrame({'shz':shz}))
                df = df.join(pd.DataFrame({'sv':sv}))
                df = df.join(pd.DataFrame({'srtk':srtk}))
                df = df.join(pd.DataFrame({'se':se}))
                output.write(','.join([str(i) for i in [alg,thresh,surface,trial_num,hz_mean,points,
                                                        RMSE, RMSE_raw,','.join([str(f) for f in RMSE_g]),
                                                        P95, P95_raw, ','.join([str(f) for f in P95_g]),
                                                        nans, slope, r_value, p_value, t_test]] + ['\n']))
                # Sort by algorithm
                if alg == 'USURFEx':
                    d_usurfex[thresh].update({trial : df})
                elif alg == 'USURFEx_N2':
                    d_usurfex_n2[thresh].update({trial : df})
                elif alg == 'SURFEx':
                    d_usurf[thresh].update({trial : df})
                elif alg == 'SURFEx_N2':
                    d_usurf_n2[thresh].update({trial : df})
                elif alg == 'ORB_HAMMING':
                    d_orb_hamming[thresh].update({trial : df})
                elif alg == 'ORB_HAMMING_N2':
                    d_orb_hamming_n2[thresh].update({trial : df})
                elif alg == 'ORB_HAMMINGCL':
                    d_orb_hamming2[thresh].update({trial : df})
                elif alg == 'ORB_HAMMINGCL_N2':
                    d_orb_hamming2_n2[thresh].update({trial : df})
                elif alg == 'SIFT':
                    d_sift[thresh].update({trial : df})
                elif alg == 'SIFT_N2':
                    d_sift_n2[thresh].update({trial : df})
                ## elif alg == 'ORB_FAST':
                ##     d_orb_fast[thresh].update({trial : df})
                ## elif alg == 'ORB_FAST_N2':
                ##     d_orb_fast_n2[thresh].update({trial : df})
                ## elif alg == 'ORB_L2':
                ##     d_orb_l2[thresh].update({trial : df})
                else:
                    raise Exception("Bad algorithm: %s-%s" % (alg, thresh))
                
                # Sort by surface
                if surface == 'corn':
                    d_corn[alg].append(df)
                elif surface == 'hay':
                    d_hay[alg].append(df)
                elif surface == 'grass':
                    d_grass[alg].append(df)
                elif surface == 'residue':
                    d_residue[alg].append(df)
                elif surface == 'asphault':
                    d_asphault[alg].append(df)
                elif surface == 'gravel':
                    d_gravel[alg].append(df)
                else:
                    raise Exception("Bad surface: %s" % surface)

    ## Figure #1
    print("===============================================================")
    try:
        # Good Example
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        fig.add_subplot(1,3,1)
        trial = 'asphault-3'
        key = 'sv'
        plt.plot(d_orb_hamming2[ORB_TREATMENTS[0]][trial][key], c=COLORS['ORB_HAMMING2'])
        plt.plot(d_usurf[SURF_TREATMENTS[0]][trial]['srtk'], c=COLORS['RTK'], linestyle='dotted', linewidth=4)
        plt.ylim([1,5.5])
        plt.yticks([1,2,3,4,5],fontsize=14)
        plt.ylabel("Travel Speed (m/s)", fontsize=14)
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.title("Asphalt, Trial #3")

        # Bad Example 
        fig.add_subplot(1,3,2)
        trial = 'corn-5'
        key = 'sv'
        line_cv, = plt.plot(d_orb_hamming2[ORB_TREATMENTS[0]][trial][key], label="CLORB", c=COLORS['ORB_HAMMING2'])
        line_rtk, = plt.plot(d_usurf[SURF_TREATMENTS[0]][trial]['srtk'], label="RTK GNSS", c=COLORS['RTK'], linestyle='dotted', linewidth=4)
        plt.ylim([1,5.5])
        plt.yticks([1,2,3,4,5],fontsize=14)
        plt.title("Corn Residue, Trial #2")
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        
        # Legend
        fig.add_subplot(1,3,3)
        plt.axis('off')
        plt.legend(handles=[line_rtk, line_cv])
        fig.show()
    except Exception as e:
        print str(e)
        
    ## Figure: Linear Regressian (By Surface)
    print("===============================================================")
    try:
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        output = open('linregress.csv', 'w')
        output.write('algorithm,surface,slope,r_value,p_value,t_test,p95\n')
        # Gravel Subplot
        fig.add_subplot(3,3,1)
        surface='gravel'
        for alg, trials in d_gravel.iteritems(): # Gravel
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Asphault subplot
        fig.add_subplot(3,3,2)
        surface='asphault'
        for alg, trials in d_asphault.iteritems(): # Asphault
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Grass Subplot
        fig.add_subplot(3,3,4)
        surface='grass'
        for alg, trials in d_grass.iteritems(): # Grass
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Residue Subplot
        surface='residue'
        fig.add_subplot(3,3,5)
        for alg, trials in d_residue.iteritems(): # Residue
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Corn Subplot
        surface='corn'
        fig.add_subplot(3,3,7)
        for alg, trials in d_corn.iteritems(): # Corn
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Hay Subplot
        surface='hay'
        ax = fig.add_subplot(3,3,8)
        for alg, trials in d_hay.iteritems(): # hay
            if len(trials) == 0: print "WARNING: %s is empty!" % alg
            else:
                composite = pd.concat(trials)
                x = composite['srtk'][~np.isnan(composite['sv'])]
                y = composite['sv'][~np.isnan(composite['sv'])]
                slope, r_value, t_test, p_value = lin_fit(x,y)
                P95 = p95(composite['sv'], composite['srtk'])
                newline = "%s,%s,%f,%f,%f,%f,%f\n" % (alg,surface,slope,r_value, p_value,t_test,P95)
                output.write(newline)
                i = np.arange(1,6)
                Y = np.polyval([slope], i)
                plt.plot(i, Y, c=COLORS[alg])
                plt.scatter(x, y, c=COLORS[alg], s=1, alpha=0.1, edgecolors='none')
        plt.axis([1, 5, 1, 5])
        plt.title(SURFACE_LABELS[surface])
        # Legend
        fig.add_subplot(3,3,6)
        plt.axis('off')
        plt.legend(handles=LEGEND)
        fig.show()
    except Exception as e:
        print str(e)
        
    ## Figure: QQ-Plot (By Surface)
    print("===============================================================")
    try:
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        # Gravel Subplot
        fig.add_subplot(3,2,1)
        surface='gravel'
        for alg, trials in d_gravel.iteritems(): # Gravel
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
    except:
        pass
    try:
        # Asphault subplot
        fig.add_subplot(3,2,2)
        surface='asphault'
        for alg, trials in d_asphault.iteritems(): # Asphault
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
    except:
        pass
    try:
        # Grass Subplot
        fig.add_subplot(3,2,3)
        surface='grass'
        for alg, trials in d_grass.iteritems(): # Grass
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
        plt.ylabel('Ordered Median Error (m/s)', fontsize=14)
    except:
        pass
    try:
        # Residue Subplot
        surface='residue'
        fig.add_subplot(3,2,4)
        for alg, trials in d_residue.iteritems(): # Residue
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
    except:
        pass
    try:
        # Corn Subplot
        surface='corn'
        fig.add_subplot(3,2,5)
        for alg, trials in d_corn.iteritems(): # Corn
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
        plt.xlabel('Quantiles', fontsize=14)
    except:
        pass
    try:
        # Hay Subplot
        surface='hay'
        fig.add_subplot(3,2,6)
        for alg, trials in d_hay.iteritems(): # hay
            composite = pd.concat(trials)
            nan = np.isnan(composite['sv'])
            x = composite['sv'][~nan]
            y = composite['srtk'][~nan]
            (osm, osr), (slope, intercept, r)  = stats.probplot(x - y, dist="norm")
            plt.scatter(osm, osr, s=1, edgecolors='none', c=COLORS[alg])
        plt.axis([-4,4,-0.3,0.3])
        plt.xticks([-4,-3,-2,-1,0,1,2,3,4], fontsize=14)
        plt.yticks([-0.2,0.0,0.2], fontsize=14)
        plt.title(SURFACE_LABELS[surface])
        plt.xlabel('Quantiles', fontsize=14)
    except:
        pass
    try:
        # Legend
        #fig.add_subplot(3,3,6)
        #plt.axis('off')
        #plt.legend(handles=LEGEND)
        fig.show()
    except Exception as e:
        print str(e)
            
    ## Figure 3. Normalized Histogram (By Surface)
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    algorithm = 'ORB_HAMMINGCL'
    # Gravel
    surface = 'gravel'
    fig.add_subplot(3,2,1)
    for alg, trials in d_gravel.iteritems():
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:
            # Corrected
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.axis([-1, 1, 0, 1])
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Asphault
    surface = 'asphault'
    fig.add_subplot(3,2,2)
    for alg, trials in d_asphault.iteritems():
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:       
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.axis([-1, 1, 0, 1])
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Grass
    surface = 'grass'
    fig.add_subplot(3,2,3)
    for alg, trials in d_grass.iteritems(): 
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.axis([-1, 1, 0, 1])
    plt.title(SURFACE_LABELS[surface])
    plt.ylabel("Normalized Frequency", fontsize=14)
    # Residue
    surface = 'residue'
    fig.add_subplot(3,2,4)
    for alg, trials in d_residue.iteritems(): 
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.axis([-1, 1, 0, 1])
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Corn
    surface = 'corn'
    fig.add_subplot(3,2,5)
    for alg, trials in d_corn.iteritems(): 
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.axis([-1, 1, 0, 1])
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    plt.xlabel("Error (m/s)", fontsize=14)
    # Hay
    surface = 'hay'
    fig.add_subplot(3,2,6)
    for alg, trials in d_hay.iteritems(): 
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        elif alg == algorithm:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.plot(np.zeros(2), np.linspace(0,1,2), c='black')
    plt.axis([-1, 1, 0, 1])
    plt.xticks([-0.5, 0.0, 0.5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.xlabel("Error (m/s)", fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    fig.show()
    
    ## Figure (by Feature-Detector): Scatter of RTK vs Repeatibility
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    # SURF Variants
    t = SURF_TREATMENTS[0]
    fig.add_subplot(5,2,1)
    for trial, df in d_usurf[t].iteritems():
        plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURF'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('SURF (cross-check)')
    fig.add_subplot(5,2,2)
    for trial, df in d_usurf_n2[t].iteritems():
        plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURF_N2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('SURF (ratio-test)')
    fig.add_subplot(5,2,3)
    for trial, df in d_usurfex[t].iteritems():
        plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURFEx'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('U-SURF (cross-check)')
    fig.add_subplot(5,2,4)
    for trial, df in d_usurfex_n2[t].iteritems():
        plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURFEx_N2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('U-SURF (ratio-test)')
    # ORB Variants
    t = ORB_TREATMENTS[0]
    fig.add_subplot(5,2,5)
    for trial, df in d_orb_hamming[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.ylabel('Inlier-Outlier Ratio', fontsize=14)
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('ORB (cross-check)')          
    fig.add_subplot(5,2,6)
    for trial, df in d_orb_hamming_n2[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING_N2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('ORB (ratio-test)')               
    fig.add_subplot(5,2,7)
    for trial, df in d_orb_hamming2[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('CLORB (cross-check)') 
    fig.add_subplot(5,2,8)
    for trial, df in d_orb_hamming2_n2[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING2_N2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('CLORB (ratio-test)')
    t = SIFT_TREATMENTS[0]
    fig.add_subplot(5,2,9)
    for trial, df in d_sift[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['SIFT'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xlabel('True Speed (m/s)', fontsize=14)
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('SIFT (cross-check)')
    fig.add_subplot(5,2,10)
    for trial, df in d_sift_n2[t].iteritems():
        plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['SIFT_N2'], s=4, edgecolors='none')
    plt.axis([1, 5, 0, 1])
    plt.xlabel('True Speed (m/s)', fontsize=14)
    plt.xticks([1,2,3,4,5], fontsize=14)
    plt.yticks([0, 0.5, 1.0], fontsize=14)
    plt.title('SIFT (ratio-test)')
    fig.show()

    # Figure: SURF Variants
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    handles = []
    for t in SURF_TREATMENTS:
        fig.add_subplot(2,2,1)
        composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_surf, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('SURF (cross-check)')
        plt.ylabel('2nd-Order Best-Fit Error (m/s)', fontsize=14)
        handles.append(line_surf)
    plt.legend(handles=handles, loc=2)
    handles = []
    for t in SURF_TREATMENTS:
        fig.add_subplot(2,2,2)
        composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_surf_n2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('SURF (ratio-test)')
        handles.append(line_surf_n2)
    plt.legend(handles=handles, loc=2)
    handles = []
    for t in SURF_TREATMENTS:
        fig.add_subplot(2,2,3)
        composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_surf2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('U-SURF (cross-check)')
        plt.ylabel('2nd-Order Best-Fit Error (m/s)', fontsize=14)
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        handles.append(line_surf2)
    plt.legend(handles=handles, loc=2)
    handles = []
    for t in SURF_TREATMENTS:
        fig.add_subplot(2,2,4)
        composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_surf2_n2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('U-SURF (ratio-test)')
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        handles.append(line_surf2_n2)
    plt.legend(handles=handles, loc=2)
    fig.show()

    # ORB Variants  
    print("===============================================================")  
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    handles = []
    fig.add_subplot(2,2,1)
    for t in ORB_TREATMENTS:
        composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_orb, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.ylabel('Error (m/s)', fontsize=14)
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('ORB (cross-check)')   
        plt.ylabel('2nd-Order Best-Fit Error (m/s)', fontsize=14)
        handles.append(line_orb)
    plt.legend(handles=handles, loc=2)
    #
    handles = []     
    fig.add_subplot(2,2,2)  
    for t in ORB_TREATMENTS:
        composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_orb_n2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('ORB (ratio-test)')
        handles.append(line_orb_n2)
    plt.legend(handles=handles, loc=2)
    #
    handles = [] 
    fig.add_subplot(2,2,3)
    for t in ORB_TREATMENTS:             
        composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_orb2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('CLORB (cross-check)') 
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        plt.ylabel('2nd-Order Best-Fit Error (m/s)', fontsize=14)
        handles.append(line_orb2)
    plt.legend(handles=handles, loc=2)
    #
    handles = []
    fig.add_subplot(2,2,4)
    for t in ORB_TREATMENTS:
        composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_orb2_n2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('CLORB (ratio-test)')
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        handles.append(line_orb2_n2)
    plt.legend(handles=handles, loc=2)
    fig.show()
    
    # Figure: SIFT Variants
    print("===============================================================")  
    # SIFT (cross-check)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,2,1)
    handles = []
    for t in SIFT_TREATMENTS:
        composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_sift, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('SIFT (cross-check)')
        plt.ylabel('2nd-Order Best-Fit Error (m/s)', fontsize=14)
        handles.append(line_sift)
    plt.legend(handles=handles, loc=2)
    # SIFT (ratio-test)
    fig.add_subplot(1,2,2)
    handles = []
    for t in SIFT_TREATMENTS:
        composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
        X,Y = poly2d(composite, kx='srtk', ky='se')
        line_sift_n2, = plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))
        plt.xlabel('RTK-DGPS Speed (m/s)', fontsize=14)
        plt.axis([1, 5, 0, 0.4])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4], fontsize=14)
        plt.title('SIFT (ratio-test)')
        handles.append(line_sift_n2)
    plt.legend(handles=handles, loc=2)
    fig.show()
    
    ## Figure 7a. Barchart (by Feature-Detector): RTK-groups vs. RMSE
    print("===============================================================")
    output = open('rmse.csv', 'w')
    output.write('algorithm,hz,rmse,rmse_1,rmse_2,rmse_3,rmse_4\n')
    index = np.arange(NUM_GROUPS) + 1
    bar_width = 0.09
    opacity = 1.0
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,2,1)
    plt.axis([1, 5+bar_width, 0, 0.5])
    # SURF Variants
    t = SURF_TREATMENTS[0]
    i = 1
    alg = 'SURF_1NN'    
    composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
    RMSE, groups = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['USURF'],
            label='U-SURF')
    i = 2
    alg = 'SURF_2NN'    
    composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
    RMSE, groups = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['USURF_N2'],
            label='U-SURF (ratio-test)')
    i = 3
    alg = 'USURF_1NN'    
    composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
    RMSE, groups = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['USURFEx'],
            label='U-SURF Extended (cross-checking)')
    i = 4
    alg = 'USURF_2NN'    
    composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
    RMSE, groups = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['USURFEx_N2'],
            label='U-SURF Extended (ratio-test)')
    # ORB Variants
    t = ORB_TREATMENTS[0]
    i = 5     
    alg = 'ORB_1NN'    
    composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING'],
            label='ORB (cross-checking)')
    i = 6    
    alg = 'ORB_2NN'    
    composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING_N2'],
            label='ORB (ratio-test)')
    i = 7    
    alg = 'CLORB_1NN'   
    composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING2'],
            label='CLORB (cross-check)')
    i = 8    
    alg = 'CLORB_2NN'     
    composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING2_N2'],
            label='CLORB (ratio-test)')
    # SIFT Variants
    t = SIFT_TREATMENTS[0]
    i = 9    
    alg = 'SIFT_1NN'     
    composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['SIFT'],
            label='SIFT (cross-check)')
    i = 10  
    alg = 'SIFT_2NN'    
    composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
    RMSE, _ = rmse_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    RMSE_all, _ = rmse_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    p = np.mean(composite['p'])
    vals = [alg, p, RMSE_all[0]] + RMSE + ['\n']
    newline = ','.join([str(v) for v in vals])
    print newline
    output.write(newline)
    plt.bar(index+bar_width*i, RMSE, bar_width,
            alpha=opacity,
            color=COLORS['SIFT_N2'],
            label='SIFT (ratio-test)')
    plt.xticks([1.5,2.5,3.5,4.5], ['1.0 - 2.0', '2.0 - 3.0', '3.0 - 4.0', '4.0 - 5.0'], fontsize=14)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=14)
    plt.ylabel('RMSE (m/s)', fontsize=14)
    plt.xlabel('Speed (m/s)', fontsize=14)
    fig.show()
    output.close()

    ## Figure 7b. Barchart (by Feature-Detector): RTK-groups vs. 95th
    print("===============================================================")
    output = open('p95.csv', 'w')
    output.write('algorithm,hz,p95,p95_1,p95_2,p95_3,p95_4\n')
    fig.add_subplot(1,2,2)
    plt.axis([1, 5+bar_width, 0, 0.5])
    # SURF Variants
    t = SURF_TREATMENTS[0]
    alg = 'USURF'
    i = 1
    composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )
    i = 2
    alg = 'USURF_N2'
    composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )
    i = 3
    alg = 'USURFEx'
    composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )
    i = 4
    alg = 'USURFEx_N2'
    composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )
    # ORB Variants
    t = ORB_TREATMENTS[0]     
    alg = 'ORB_HAMMING'    
    i = 5
    composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )   
    i = 6
    alg = 'ORB_HAMMING_N2'
    composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg]
        )      
    i = 7
    alg = 'ORB_HAMMING2'
    composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg])  
    i = 8
    alg = 'ORB_HAMMING2_N2'
    composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg])
    # SIFT Variants
    t = SIFT_TREATMENTS[0]         
    i = 9
    alg = 'SIFT'
    composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg])       
    i = 10
    alg = 'SIFT_N2'
    composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
    P95, _ = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS, SPEED_RANGE)
    P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
    hz = np.mean(composite['hz'])
    vals = [alg,hz,P95_all[0]] + P95 + ['\n']
    newline = ','.join([str(v) for v in vals])
    output.write(newline)
    plt.bar(index+bar_width*i, P95, bar_width,
            alpha=opacity,
            color=COLORS[alg], hatch=HATCHES[alg],
            label=ALGORITHM_LABELS[alg])
    plt.xticks([1.5,2.5,3.5,4.5], ['1.0 - 2.0', '2.0 - 3.0', '3.0 - 4.0', '4.0 - 5.0'], fontsize=14)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=14)
    plt.ylabel('95th Percentile Error (m/s)', fontsize=14)
    plt.xlabel('Speed (m/s)', fontsize=14)
    fig.show()
    output.close()
    
    ## Figure 7c. Lines (by Feature-Detector): RTK-groups vs. 95th
    print("===============================================================")
    output = open('p95_by_threshold.csv', 'w')
    output.write('algorithm,threshold,hz,n,p95,p95_1,p95_2,p95_3,p95_4\n')
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    # SURF Variants
    handles = []  
    for t in SURF_TREATMENTS:
        alg = 'USURF'
        fig.add_subplot(2,2,1)
        composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))  
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.ylabel('Error (m/s)', fontsize=14)    
        plt.title('SURF (cross-check)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []  
    for t in SURF_TREATMENTS: 
        alg = 'USURF_N2'
        fig.add_subplot(2,2,2)
        composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))  
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)    
        plt.title('SURF (ratio-test)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []  
    for t in SURF_TREATMENTS:         
        alg = 'USURFEx'
        fig.add_subplot(2,2,3)
        composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))    
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14) 
        plt.ylabel('Error (m/s)', fontsize=14)   
        plt.xlabel('Speed (m/s)', fontsize=14)
        plt.title('U-SURF (cross-check)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []  
    for t in SURF_TREATMENTS:       
        alg = 'USURFEx_N2'
        fig.add_subplot(2,2,4)
        composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))      
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14) 
        plt.xlabel('Speed (m/s)', fontsize=14) 
        plt.title('U-SURF (ratio-test)')      
        handles.append(line)
    plt.legend(handles=handles, loc=2)   
    fig.show()
  
    # ORB Variants
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    handles = []  
    for t in ORB_TREATMENTS:     
        alg = 'ORB_HAMMING'
        fig.add_subplot(2,2,1)  
        composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))         
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.ylabel('Error (m/s)', fontsize=14)
        plt.title('ORB (cross-check)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []  
    for t in ORB_TREATMENTS:      
        alg = 'ORB_HAMMING_N2'
        fig.add_subplot(2,2,2)  
        composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz'])) 
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.title('ORB (ratio-test)')     
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []  
    for t in ORB_TREATMENTS:          
        alg = 'ORB_HAMMING2'
        fig.add_subplot(2,2,3)  
        composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz'])) 
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14) 
        plt.xlabel('Speed (m/s)', fontsize=14)  
        plt.ylabel('Error (m/s)', fontsize=14)
        plt.title('CLORB (cross-check)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []    
    for t in ORB_TREATMENTS:         
        alg = 'ORB_HAMMING2_N2'
        fig.add_subplot(2,2,4)  
        composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))  
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.xlabel('Speed (m/s)', fontsize=14)    
        plt.title('CLORB (ratio-test)')    
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    fig.show()

    # SIFT Variants
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    handles = []
    for t in SIFT_TREATMENTS:         
        alg = 'SIFT'
        fig.add_subplot(2,2,3)
        composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))         
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.ylabel('Error (m/s)', fontsize=14)
        plt.xlabel('Speed (m/s)', fontsize=14)
        plt.title('SIFT (cross-check)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    handles = []
    for t in SIFT_TREATMENTS:           
        alg = 'SIFT_N2'
        fig.add_subplot(2,2,4)
        composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
        P95, groups = p95_by_group(composite['sv'], composite['srtk'], NUM_GROUPS*SUB_GROUPS, SPEED_RANGE)
        P95_all, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        vals = [alg,t,hz,P95_all[0]] + P95 + ['\n']
        newline = ','.join([str(v) for v in vals])
        output.write(newline)
        line, = plt.plot(groups, P95, c='black', linestyle=LINE_TYPES[t], label=t+' (%2.1f Hz)' % np.nanmean(composite['hz']))           
        plt.axis([1, 5, 0, 0.6])
        plt.xticks([1,2,3,4,5], fontsize=14)
        plt.yticks([0, 0.2, 0.4, 0.6], fontsize=14)  
        plt.xlabel('Speed (m/s)', fontsize=14)
        plt.title('SIFT (ratio-test)')
        handles.append(line)
    plt.legend(handles=handles, loc=2)     
    fig.show()
    output.close()

    ## Figure 7d. Lines (by Feature-Detector): Hz vs. 95th
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,1,1)
    # SURF Variants
    alg = 'USURF'
    hz_all = []
    P95_all = []
    for t in SURF_TREATMENTS:
        composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'USURF_N2'
    hz_all = []
    P95_all = []
    for t in SURF_TREATMENTS: 
        composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'USURFEx'
    hz_all = []
    P95_all = []
    for t in SURF_TREATMENTS:         
        composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'USURFEx_N2'
    hz_all = []
    P95_all = []
    for t in SURF_TREATMENTS:       
        composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    # ORB Variants
    alg = 'ORB_HAMMING' 
    hz_all = []
    P95_all = []
    for t in ORB_TREATMENTS:     
        composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'ORB_HAMMING_N2'
    hz_all = []
    P95_all = []
    for t in ORB_TREATMENTS:      
        composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'ORB_HAMMING2'
    hz_all = []
    P95_all = []
    for t in ORB_TREATMENTS:          
        composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'ORB_HAMMING2_N2'
    hz_all = []
    P95_all = []
    for t in ORB_TREATMENTS:         
        composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    # SIFT Variants
    alg = 'SIFT'
    hz_all = []
    P95_all = []
    for t in SIFT_TREATMENTS:         
        composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz = np.mean(composite['hz'])
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    alg = 'SIFT_N2'
    hz_all = []
    P95_all = []
    for t in SIFT_TREATMENTS:           
        composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
        P95, _ = p95_by_group(composite['sv'], composite['srtk'], 1, SPEED_RANGE)
        hz_all.append(hz)
        P95_all.append(P95)
    plt.scatter(hz_all, P95_all, c=COLORS[alg], marker=MARKERS[alg], s=MARKER_SIZE, label=ALGORITHM_LABELS[alg])
    plt.axis([0, 60, 0.2, 0.3])
    plt.xticks([10,20,30,40,50,60], fontsize=14)
    plt.yticks([0.2, 0.22, 0.24, 0.26, 0.28, 0.3], fontsize=14)  
    plt.xlabel('Processing Rate (Hz)', fontsize=14)
    plt.ylabel('95th Percentile Error (m/s)', fontsize=14)
    plt.legend(scatterpoints=1, loc=4)
    fig.show()

    ## Figure: Percent Error (By Surface) for a specific algorithm(s)
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    targets = ['ORB_HAMMINGCL']
    a = 1.0
    # Gravel Subplot
    fig.add_subplot(3,2,1)
    surface='gravel'
    for alg, trials in d_gravel.iteritems(): # Gravel
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Asphault subplot
    fig.add_subplot(3,2,2)
    surface='asphault'
    for alg, trials in d_asphault.iteritems(): # Asphault
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Grass Subplot
    fig.add_subplot(3,2,3)
    surface='grass'
    for alg, trials in d_grass.iteritems(): # Grass
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    plt.ylabel('Percent Error (%)', fontsize=14)
    # Residue Subplot
    surface='residue'
    fig.add_subplot(3,2,4)
    for alg, trials in d_residue.iteritems(): # Residue
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    # Corn Subplot
    surface='corn'
    fig.add_subplot(3,2,5)
    for alg, trials in d_corn.iteritems(): # Corn
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    plt.xlabel(r'Acceleration $\mathregular{(m/s^{2})}}$', fontsize=14)
    # Hay Subplot
    surface='hay'
    fig.add_subplot(3,2,6)
    for alg, trials in d_hay.iteritems(): # hay
        if alg in targets:
            for df in trials:
                plt.scatter(df['drtk'], df['d'], c=COLORS[alg], s=1, edgecolors='none', alpha=a)
    plt.axis([-2.5, 2.5, -20, 20])
    plt.xticks([-2.0, -1.0, 0.0, 1.0, 2.0], fontsize=14)
    plt.yticks([-20, 0, 20], fontsize=14)
    plt.title(SURFACE_LABELS[surface])
    plt.xlabel('Acceleration $\mathregular{(m/s^{2})}}$', fontsize=14)
    fig.show()
    
    # Figure (Threshold x Feature-Detector): RTK vs Hz
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    # SURF Variants
    for t in SURF_TREATMENTS:
        fig.add_subplot(5,2,1)
        alg = 'SURF'
        composite = pd.concat([df for trial, df in d_usurf[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('SURF (cross-check)')
        fig.add_subplot(5,2,2)
        alg = 'SURF_N2'
        composite = pd.concat([df for trial, df in d_usurf_n2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('SURF (ratio-test)')
        fig.add_subplot(5,2,3)
        alg = 'USURFEx'
        composite = pd.concat([df for trial, df in d_usurfex[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('U-SURF Extended (cross-check)')
        fig.add_subplot(5,2,4)
        alg = 'USURFEx_N2'
        composite = pd.concat([df for trial, df in d_usurfex_n2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('U-SURF Extended (ratio-test)')
    # ORB Variants
    for t in ORB_TREATMENTS:        
        fig.add_subplot(5,2,5)
        alg = 'ORB_HAMMING'
        composite = pd.concat([df for trial, df in d_orb_hamming[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('ORB (cross-check)')   
        fig.add_subplot(5,2,6)
        alg = 'ORB_HAMMING_N2'
        composite = pd.concat([df for trial, df in d_orb_hamming_n2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('ORB (ratio-test)')       
        fig.add_subplot(5,2,7)
        alg = 'ORB_HAMMINGCL'
        composite = pd.concat([df for trial, df in d_orb_hamming2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('CLORB (cross-check)')     
        fig.add_subplot(5,2,8)
        alg = 'ORB_HAMMINGCL_N2'
        composite = pd.concat([df for trial, df in d_orb_hamming2_n2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('CLORB (ratio-test)')
    # SIFT Variants
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(5,2,9)
        alg = 'SIFT'
        composite = pd.concat([df for trial, df in d_sift[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('SIFT (cross-check)')      
        fig.add_subplot(5,2,10)
        alg = 'SIFT_N2'
        composite = pd.concat([df for trial, df in d_sift_n2[t].iteritems()])
        X, Y = poly2d(composite, kx='rtk', ky='hz')
        plt.plot(X, Y, c='black', linestyle=LINE_TYPES[t])
        plt.axis([1, 5, 0, 50])
        plt.title('SIFT (ratio-test)')
    fig.show()

    # Figure (by Feature-Detector): Hz vs N-features (best-of)
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,1,1)
    # SURF Variants
    t = SURF_TREATMENTS[0]
    for trial, df in d_usurf[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['USURF'], s=1, edgecolors='none', alpha=0.2)
    for trial, df in d_usurf_n2[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['USURF_N2'], s=1, edgecolors='none', alpha=0.2)
    for trial, df in d_usurfex[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['USURFEx'], s=1, edgecolors='none', alpha=0.2)
    for trial, df in d_usurfex_n2[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['USURFEx_N2'], s=1, edgecolors='none', alpha=0.2)
    # ORB Variants
    t = ORB_TREATMENTS[0]     
    for trial, df in d_orb_hamming[t].iteritems():
        plt.scatter(df['p'], df['hz'], c=COLORS['ORB_HAMMING'], s=1, edgecolors='none', alpha=0.2)   
    for trial, df in d_orb_hamming_n2[t].iteritems():
        plt.scatter(df['p'], df['hz'], c=COLORS['ORB_HAMMING_N2'], s=1, edgecolors='none', alpha=0.2)      
    for trial, df in d_orb_hamming2[t].iteritems():
        plt.scatter(df['p'], df['hz'], c=COLORS['ORB_HAMMING2'], s=1, edgecolors='none', alpha=0.2)       
    for trial, df in d_orb_hamming2_n2[t].iteritems():
        plt.scatter(df['p'], df['hz'], c=COLORS['ORB_HAMMING2_N2'], s=1, edgecolors='none', alpha=0.2)
    # SIFT Variants
    t = SIFT_TREATMENTS[0]        
    for trial, df in d_sift[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['SIFT'], s=1, edgecolors='none', alpha=0.2)       
    for trial, df in d_sift_n2[t].iteritems():
       plt.scatter(df['p'], df['hz'], c=COLORS['SIFT_N2'], s=1, edgecolors='none', alpha=0.2)
    plt.axis([0, 500, 0, 60])
    plt.ylabel('Processing Time (Hz)', fontsize = 14)
    plt.yticks([0,20,40,60], fontsize = 14)
    plt.xlabel('Valid Matches', fontsize = 14)
    plt.xticks([0,250,500], fontsize = 14)
    #plt.legend(handles=LEGEND)
    fig.show()

    ## Plot all and wait...
    plt.waitforbuttonpress()
