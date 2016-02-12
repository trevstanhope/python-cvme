import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as sig
import time
import scipy.stats as stats

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
def moving_median(interval, window_size):
    return sig.medfilt(interval, kernel_size=window_size)
def rmse(predictions, targets):
    return np.sqrt(np.nanmean(((np.array(predictions) - np.array(targets)) ** 2)))
def p95(predictions, targets):
    if len(predictions) == 0 or len(targets) == 0:
        return np.NAN
    else:
        abs_err = np.abs(np.array(predictions) - np.array(targets))
        return np.percentile(abs_err[~np.isnan(abs_err)], 95)
def p95_by_group(y, x, bins, lims):
    offset = (lims[1] - lims[0]) / bins
    groups_a = np.linspace(lims[0], lims[1] - offset, num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1], num=bins)
    groups = zip(groups_a, groups_b)
    P95 = []
    for (a,b) in groups:
        Y = y[np.logical_and(y>a, y<=b)]
        X = x[np.logical_and(y>a, y<=b)]
        v = p95(Y, X)
        if np.isnan(v): print "NAN WARNING in RMSE by Group!"
        P95.append(v) 
    return P95, groups_b
def rmse_by_group(y, x, bins, lims):
    offset = (lims[1] - lims[0]) / bins
    groups_a = np.linspace(lims[0], lims[1] - offset, num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1], num=bins)
    groups = zip(groups_a, groups_b)
    RMSE = []
    for (a,b) in groups:
        Y = y[np.logical_and(y>a, y<=b)]
        X = x[np.logical_and(y>a, y<=b)]
        v = rmse(Y, X)
        if np.isnan(v): print "NAN WARNING in RMSE by Group!"
        RMSE.append(v)
    return RMSE, groups_b
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
    SMOOTHING_WINDOW = 9
    ORB_TREATMENTS = ['500']
    SURF_TREATMENTS = ['2000']
    SIFT_TREATMENTS = ['500']
    SURFACES = ['asphault', 'grass', 'gravel', 'residue', 'corn', 'hay']
    ALGORITHMS = ['USURF',
                  'USURF_N2',
                  'USURFEx',
                  'USURFEx_N2',
                  'ORB_HAMMING',
                  'ORB_HAMMING_N2',
                  'ORB_HAMMINGCL',
                  'ORB_HAMMINGCL_N2',
                  'SIFT'
                  ]
    DEZOOM = 1.01
    CORR_FACTORS = {
        'asphault': 1.00,
        'gravel': 1.00,
        'residue': 0.99,
        'grass': 0.96,
        'hay': 0.93,
        'corn' : 0.96,
        'soy' : 0.65
    }
    TRIAL_BLACKLIST = ['gravel-1',
                       'corn-1', 'corn-3', 'corn-4', 'corn-7', 'corn-8', 'corn-11',
                       'hay-1']
    HEADERS = ['rtk','v', 't', 'm', 'p', 'n'] # Column headers (must match the .csv headers!)
    TREATMENTS = [tuple(d.split('-')) for d in os.listdir(".") if os.path.isdir(d)]
    TRIALS = [f.split('/')[-1].split('.')[0] for f in glob.glob("../data/*.csv")]
    COLORS = {
        "USURF" : "red",
        "USURF_N2" : "darkred",
        "USURFEx" : "goldenrod",
        "USURFEx_N2" : "darkgoldenrod",
        "ORB_HAMMING" : "green",
        "ORB_HAMMING_N2" : "darkgreen",
        "ORB_HAMMING2" : "blue",
        "ORB_HAMMING2_N2" : "darkblue",
        "ORB_HAMMINGCL" : "magenta",
        "ORB_HAMMINGCL_N2" : "darkmagenta",
        "ORB_L2" : "orange",
        "ORB_L2_N2" : "darkorange",
        "ORB_FAST" : "cyan",
        "ORB_FAST_N2" : "darkcyan",
        "SIFT" : "gray",
        "SIFT_N2" : "dimgray"
    }
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
    print("===============================================================")
    output = open("summary.csv", "w")
    output.write("alg,thresh,surface,trial_num,rmse,p95,nans\n")
    print("alg,thresh,surface,trial_num,hz,points,rmse,p95,nans")
    for alg,thresh in TREATMENTS:
        for f in glob.glob(os.path.join(alg + '-' + thresh,'*.csv')):
            trial = f.split('/')[-1].split('.')[0]
            surface = trial.split('-')[0]
            trial_num = trial.split('-')[1]
            if (surface in SURFACES) and (alg in ALGORITHMS) and not (trial in TRIAL_BLACKLIST):
                df = pd.DataFrame.from_csv(f)
                rtk = df['rtk'] / 3.6
                hz = df['hz']
                v = df['v'] * DEZOOM / 3.6
                nans = np.count_nonzero(np.isnan(v))
                cv = v * CORR_FACTORS[surface]
                mask = np.isnan(cv)
                cv[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), cv[~mask])
                shz = sig.medfilt(hz, kernel_size=SMOOTHING_WINDOW) 
                srtk = sig.savgol_filter(rtk, SMOOTHING_WINDOW, 1)
                sv = sig.medfilt(cv, kernel_size=SMOOTHING_WINDOW)
                h, b = histogram_by_lims(v[~np.isnan(v)], rtk[~np.isnan(v)], SPEED_RANGE, 100, [-1,1])
                sh, sb = histogram_by_lims(sv, srtk, SPEED_RANGE, 100, [-1,1])
                RMSE, groups = rmse_by_group(sv, srtk, NUM_GROUPS, SPEED_RANGE)
                hz_mean = np.mean(hz)
                points = len(hz)
                df_mod = pd.DataFrame()
                df = df.join(pd.DataFrame({'h':h}))
                df = df.join(pd.DataFrame({'b':b}))
                df = df.join(pd.DataFrame({'sh':sh}))
                df = df.join(pd.DataFrame({'sb':sb}))
                df = df.join(pd.DataFrame({'cv':cv}))
                df = df.join(pd.DataFrame({'shz':shz}))
                df = df.join(pd.DataFrame({'sv':sv}))
                df = df.join(pd.DataFrame({'srtk':srtk}))
                df = df.join(pd.DataFrame({'srmse':RMSE}))
                print("%s,%s,%s,%s,%2.2f,%d,%2.2f,%2.2f,%d") % (alg,
                                                             thresh,
                                                             surface,
                                                             trial_num,
                                                             hz_mean,
                                                             points,
                                                             rmse(sv,srtk),
                                                             p95(sv,srtk),
                                                             nans)
                output.write(','.join([str(i) for i in [alg,
                                                        thresh,
                                                        surface,
                                                        trial_num,
                                                        hz_mean,
                                                        points,
                                                        rmse(sv,srtk),
                                                        p95(sv,srtk),
                                                        nans]] + ['\n']))
                # Sort by algorithm
                if alg == 'USURFEx':
                    d_usurfex[thresh].update({trial : df})
                elif alg == 'USURFEx_N2':
                    d_usurfex_n2[thresh].update({trial : df})
                elif alg == 'USURF':
                    d_usurf[thresh].update({trial : df})
                elif alg == 'USURF_N2':
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
    # Good Example
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,2,1)
    trial = 'asphault-2'
    key = 'sv'
    plt.plot(d_usurf[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurfex[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurf_n2[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurfex_n2[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming_n2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming2_n2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_sift[SIFT_TREATMENTS[0]][trial][key])
    plt.ylim([1,5])
    plt.title("Asphault")
    
    # Bad Example 
    fig.add_subplot(1,2,2)
    trial = 'hay-4'
    key = 'sv'
    plt.plot(d_usurf[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurfex[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurf_n2[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_usurfex_n2[SURF_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming_n2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_orb_hamming2_n2[ORB_TREATMENTS[0]][trial][key])
    plt.plot(d_sift[SIFT_TREATMENTS[0]][trial][key])
    plt.ylim([1,5])
    fig.show()
    plt.title("Corn")
    
    ## Figure: Linear Regressian (By Surface)
    print("===============================================================")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    # Gravel Subplot
    fig.add_subplot(3,2,1)
    surface='gravel'
    for alg, trials in d_gravel.iteritems(): # Gravel
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Gravel')
    # Asphault subplot
    fig.add_subplot(3,2,2)
    surface='asphault'
    for alg, trials in d_asphault.iteritems(): # Asphault
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Asphault')
    # Grass Subplot
    fig.add_subplot(3,2,3)
    surface='grass'
    for alg, trials in d_grass.iteritems(): # Grass
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Grass')
    # Residue Subplot
    surface='residue'
    fig.add_subplot(3,2,4)
    for alg, trials in d_residue.iteritems(): # Residue
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Residue')
    # Corn Subplot
    surface='corn'
    fig.add_subplot(3,2,5)
    for alg, trials in d_corn.iteritems(): # Corn
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Corn')
    # Hay Subplot
    surface='hay'
    fig.add_subplot(3,2,6)
    for alg, trials in d_hay.iteritems(): # hay
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            composite = pd.concat(trials)
            x = composite['srtk'][~np.isnan(composite['sv'])]
            y = composite['sv'][~np.isnan(composite['sv'])]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            print("%s,%s,%f,%f,%f,%f,%f" % (alg,surface,slope,intercept,r_value,p_value,std_err))
            i = np.arange(1,6)
            Y = np.polyval([slope, intercept], i)
            plt.plot(i, Y, c=COLORS[alg])
    plt.axis([1, 5, 1, 5])
    plt.title('Hay')
    fig.show()

    ## Figure 3. (By Surface)
    df_surface = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(3,2,1)
    for alg, trials in d_gravel.iteritems(): # Gravel
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            # Corrected
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Gravel')
    fig.add_subplot(3,2,2)
    for alg, trials in d_asphault.iteritems(): # Asphault
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:        
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Asphault')
    fig.add_subplot(3,2,3)
    for alg, trials in d_grass.iteritems(): # Grass
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Grass')
    fig.add_subplot(3,2,4)
    for alg, trials in d_residue.iteritems(): # Residue
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Residue')
    fig.add_subplot(3,2,5)
    for alg, trials in d_corn.iteritems(): # Corn
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Corn')
    fig.add_subplot(3,2,6)
    for alg, trials in d_hay.iteritems(): # hay
        if len(trials) == 0: print "WARNING: %s is empty!" % alg
        else:
            h = [df['sh'][~np.isnan(df['sh'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['sb'][~np.isnan(df['sb'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg])
            # Raw
            h = [df['h'][~np.isnan(df['h'])] for df in trials]
            h_sum = h[0]
            for h_i in h[1:]:
                h_sum = np.add(h_sum, h_i)
            b = df['b'][~np.isnan(df['b'])][:-1]
            plt.plot(b, normalize(h_sum), c=COLORS[alg], linestyle='dashed')
    plt.axis([-1, 1, 0, 1])
    plt.title('Hay')
    fig.show()
    #df_surface.to_csv()

    # Figure (by Feature-Detector): Scatter of RTK vs Repeatibility
    df_rmse = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,1)
        for trial, df in d_usurf[t].iteritems():
            plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURF'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('USURF')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,2)
        for trial, df in d_usurf_n2[t].iteritems():
            plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURF_N2'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('USURF_N2')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,3)
        for trial, df in d_usurfex[t].iteritems():
            plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURFEx'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('USURFEx')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,4)
        for trial, df in d_usurfex_n2[t].iteritems():
            plt.scatter(df['srtk'], df['p'] / df['m'], c=COLORS['USURFEx_N2'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('USURFEx_N2')    
    for t in ORB_TREATMENTS:
        fig.add_subplot(4,2,5)
        for trial, df in d_orb_hamming[t].iteritems():
            plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('ORB_HAMMING')          
    for t in ORB_TREATMENTS:
        fig.add_subplot(4,2,6)
        for trial, df in d_orb_hamming_n2[t].iteritems():
            plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING_N2'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('ORB_HAMMING_N2')          
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,7)
        for trial, df in d_orb_hamming2[t].iteritems():
            plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING2'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('ORB_HAMMING2')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,8)
        for trial, df in d_orb_hamming2_n2[t].iteritems():
            plt.scatter(df['rtk'], df['p'] / df['m'], c=COLORS['ORB_HAMMING2_N2'], s=4, edgecolors='none')
        plt.axis([1, 5, 0, 1])
        plt.title('ORB_HAMMING2_N2')
    fig.show()
    
    # Figure (by Feature-Detector): RTK-groups vs. RMSE
    index = np.arange(NUM_GROUPS) + 1
    bar_width = 0.10
    opacity = 1.0
    df_rmse = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,1,1)
    plt.axis([1, 5, 0, 0.5])
    t = SURF_TREATMENTS[0]
    RMSE_total = []
    for trial, df in d_usurf[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['USURF'],
            label='U-SURF')
    t = SURF_TREATMENTS[0]
    RMSE_total = []
    for trial, df in d_usurf_n2[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['USURF_N2'],
            label='U-SURF (ratio-test)')
    t = SURF_TREATMENTS[0]
    RMSE_total = []
    for trial, df in d_usurfex[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*2, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['USURFEx'],
            label='U-SURF Extended (cross-checking)')
    t = SURF_TREATMENTS[0]
    RMSE_total = []
    for trial, df in d_usurfex_n2[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*3, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['USURFEx_N2'],
            label='U-SURF Extended (ratio-test)')
    t = ORB_TREATMENTS[0]      
    RMSE_total = []
    for trial, df in d_orb_hamming[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*4, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING'],
            label='ORB Hamming (cross-checking)')
    t = ORB_TREATMENTS[0]        
    RMSE_total = []
    for trial, df in d_orb_hamming_n2[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*5, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING_N2'],
            label='ORB Hamming (ratio-test)')
    t = ORB_TREATMENTS[0]        
    RMSE_total = []
    for trial, df in d_orb_hamming2[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)    
    plt.bar(index+bar_width*6, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING2'],
            label='ORB Hamming2 (cross-check)')
    t = ORB_TREATMENTS[0]        
    RMSE_total = []
    for trial, df in d_orb_hamming2_n2[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*7, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['ORB_HAMMING2_N2'],
            label='ORB Hamming2 (ratio-test)')
    t = SIFT_TREATMENTS[0]        
    RMSE_total = []
    for trial, df in d_sift[t].iteritems():
        RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
        RMSE_total.append(RMSE)
    RMSE_avg = np.mean(np.array(RMSE_total), axis=0)
    plt.bar(index+bar_width*8, RMSE_avg, bar_width,
            alpha=opacity,
            color=COLORS['SIFT'],
            label='SIFT (cross-check)')
    fig.show()
    
    ## Figure 4. (by Feature-Detector): RTK-groups vs. 95th
    index = np.arange(NUM_GROUPS)+1
    bar_width = 0.10
    opacity = 1.0
    df_p95 = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,1,1)
    plt.axis([1, 5, 0, 0.5])
    for t in SURF_TREATMENTS:
        P95_total = []
        for trial, df in d_usurf[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['USURF'],
                label='U-SURF (cross-check)')
    for t in SURF_TREATMENTS:
        P95_total = []
        for trial, df in d_usurf_n2[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*1, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['USURF_N2'],
                label='U-SURF (ratio-test)')
    for t in SURF_TREATMENTS:
        i = 2
        P95_total = []
        for trial, df in d_usurfex[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['USURFEx'],
                label='U-SURFEx (cross-check)')
    for t in SURF_TREATMENTS:
        i = 3
        P95_total = []
        for trial, df in d_usurfex_n2[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['USURFEx_N2'],
                label='U-SURFEx (cross-check)')
    for t in ORB_TREATMENTS:          
        i = 4
        P95_total = []
        for trial, df in d_orb_hamming[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['ORB_HAMMING'],
                label='ORB Hamming (cross-check)')
    for t in ORB_TREATMENTS:          
        i = 5
        P95_total = []
        for trial, df in d_orb_hamming_n2[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['ORB_HAMMING_N2'],
                label='ORB Hamming (ratio-test)')
    for t in ORB_TREATMENTS:          
        i = 6
        P95_total = []
        for trial, df in d_orb_hamming2[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['ORB_HAMMING2'],
                label='ORB Hamming2 (cross-check)')
    for t in ORB_TREATMENTS:          
        i = 7
        P95_total = []
        for trial, df in d_orb_hamming2_n2[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['ORB_HAMMING2_N2'],
                label='ORB Hamming2 (ratio-test)')
    for t in SIFT_TREATMENTS:          
        i = 8
        P95_total = []
        for trial, df in d_sift[t].iteritems():
            P95, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            P95_total.append(P95)
        P95_avg = np.mean(np.array(P95_total), axis=0)
        plt.bar(index+bar_width*i, P95_avg, bar_width,
                alpha=opacity,
                color=COLORS['SIFT'],
                label='SIFT (cross-check)')
    fig.show()
    
    # Figure (by Feature-Detector): RTK vs Hz
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,1)
        for trial, df in d_usurf[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='red', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,2)
        for trial, df in d_usurf_n2[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='darkred', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,3)
        for trial, df in d_usurfex[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='goldenrod', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
    for t in SURF_TREATMENTS:
        fig.add_subplot(4,2,4)
        for trial, df in d_usurfex_n2[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='darkgoldenrod', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,5)
        for trial, df in d_orb_hamming[t].iteritems():
            plt.scatter(df['rtk'], df['hz'], c='green', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('ORB (cross-check)')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,6)
        for trial, df in d_orb_hamming_n2[t].iteritems():
            plt.scatter(df['rtk'], df['hz'], c='darkgreen', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('ORB (ratio-test)')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,7)
        for trial, df in d_orb_hamming2[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='cyan', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('ORB Hamming2 (cross-check)')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(4,2,8)
        for trial, df in d_orb_hamming2_n2[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='darkcyan', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('ORB Hamming2 (ratio-test)')
    fig.show()

    # Figure (by Feature-Detector): Hz vs N-features
    df_robust = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(1,1,1)
        for trial, df in d_usurf[t].iteritems():
           plt.scatter(df['m'], df['hz'], c=COLORS['USURF'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in SURF_TREATMENTS:
        fig.add_subplot(1,1,1)
        for trial, df in d_usurf_n2[t].iteritems():
           plt.scatter(df['m'], df['hz'], c=COLORS['USURF_N2'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in SURF_TREATMENTS:
        fig.add_subplot(1,1,1)
        for trial, df in d_usurfex[t].iteritems():
           plt.scatter(df['m'], df['hz'], c=COLORS['USURFEx'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in SURF_TREATMENTS:
        fig.add_subplot(1,1,1)
        for trial, df in d_usurfex_n2[t].iteritems():
           plt.scatter(df['m'], df['hz'], c=COLORS['USURFEx_N2'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in ORB_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_orb_hamming[t].iteritems():
            plt.scatter(df['m'], df['hz'], c=COLORS['ORB_HAMMING'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in ORB_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_orb_hamming_n2[t].iteritems():
            plt.scatter(df['m'], df['hz'], c=COLORS['ORB_HAMMING_N2'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in ORB_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_orb_hamming2[t].iteritems():
            plt.scatter(df['m'], df['hz'], c=COLORS['ORB_HAMMING2'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in ORB_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_orb_hamming2_n2[t].iteritems():
            plt.scatter(df['m'], df['hz'], c=COLORS['ORB_HAMMING2_N2'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_sift[t].iteritems():
           plt.scatter(df['m'], df['hz'], c=COLORS['SIFT'], s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
    fig.show()
    #df_robust.to_csv()

    # Plot all and wait...
    plt.waitforbuttonpress()
