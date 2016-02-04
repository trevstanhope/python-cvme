import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as sig
import time

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
    offset = (lims[1] - lims[0]) / float(bins)
    groups_a = np.linspace(lims[0], lims[1], num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1] + offset, num=bins)
    groups = zip(groups_a, groups_b)
    return [p95(y[np.logical_and(y>a, y<=b)], x[np.logical_and(y>a, y<=b)]) for (a,b) in groups], groups_a
def rmse_by_group(y, x, bins, lims):
    offset = (lims[1] - lims[0]) / float(bins)
    groups_a = np.linspace(lims[0], lims[1], num=bins)
    groups_b = np.linspace(lims[0] + offset, lims[1] + offset, num=bins)
    groups = zip(groups_a, groups_b)
    return [rmse(y[np.logical_and(y>a, y<=b)], x[np.logical_and(y>a, y<=b)]) for (a,b) in groups], groups_a
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
    NUM_GROUPS = 4
    SMOOTHING_WINDOW = 7
    ORB_TREATMENTS = ['500']
    SURF_TREATMENTS = ['1000']
    SIFT_TREATMENTS = ['500']
    SURFACES = ['asphault', 'grass', 'gravel', 'residue', 'corn', 'hay']
    ALGORITHMS = ['ORB_HAMMING', 'SURF', 'SIFT']
    CORR_FACTORS = {
        'asphault': 1.01,
        'gravel': 1.01,
        'residue': 1.0,
        'grass': 0.965,
        'hay': 0.94,
        'corn' : 0.96,
        'soy' : 0.65
    }
    TRIAL_BLACKLIST = ['gravel-1',
                       'corn-1', 'corn-3', 'corn-7', 'corn-8', 'corn-9', 'corn-10',
                       'hay-1']
    HEADERS = ['rtk','v', 't', 'm', 'p', 'n'] # Column headers (must match the .csv headers!)
    TREATMENTS = [tuple(d.split('-')) for d in os.listdir(".") if os.path.isdir(d)]
    TRIALS = [f.split('/')[-1].split('.')[0] for f in glob.glob("../data/*.csv")]
    COLORS = {
        "SURF" : "r",
        "SURF2" : "y",
        "ORB_HAMMING" : "g",
        "ORB" : "c",
        "SIFT" : "b"
    }
    # Make dictionaries to load csv-files into DataFrames by algorithm
    # d_surf = {
    #   1000 : {
    #     asphault-1 : <df>
    #     ...
    #   }
    # }
    d_surf = { thresh:{} for alg,thresh in TREATMENTS}
    d_orb = { thresh:{} for alg,thresh in TREATMENTS}
    d_sift = { thresh:{} for alg,thresh in TREATMENTS}

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
    outfile = open('Summary.csv"', 'w')
    for alg,thresh in TREATMENTS:
        for f in glob.glob(os.path.join(alg + '-' + thresh,'*.csv')):
            trial = f.split('/')[-1].split('.')[0]
            surface = trial.split('-')[0]
            trial_num = trial.split('-')[1]
            if (surface in SURFACES) and (alg in ALGORITHMS) and not (trial in TRIAL_BLACKLIST):
                df = pd.DataFrame.from_csv(f)
                cv = df['v'] * CORR_FACTORS[surface] / 3.6
                shz = sig.medfilt(df['hz'], kernel_size=SMOOTHING_WINDOW) 
                srtk = sig.savgol_filter(df['rtk'] / 3.6, SMOOTHING_WINDOW, 1)
                sv = sig.medfilt(cv, kernel_size=SMOOTHING_WINDOW)
                h, b = histogram_by_lims(df['v'] / 3.6, df['rtk'] / 3.6, [1,5], 100, [-1,1])
                sh, sb = histogram_by_lims(sv, srtk, [1,5], 100, [-1,1])
                RMSE, groups = rmse_by_group(sv, srtk, NUM_GROUPS, SPEED_RANGE)
                df = df.join(pd.DataFrame({'h':h}))
                df = df.join(pd.DataFrame({'b':b}))
                df = df.join(pd.DataFrame({'sh':sh}))
                df = df.join(pd.DataFrame({'sb':sb}))
                df = df.join(pd.DataFrame({'cv':cv}))
                df = df.join(pd.DataFrame({'shz':shz}))
                df = df.join(pd.DataFrame({'sv':sv}))
                df = df.join(pd.DataFrame({'srtk':srtk}))
                df = df.join(pd.DataFrame({'srmse':RMSE}))
                print("%s\t%s\t\t%s\t%2.1f") % (alg, surface, trial_num, rmse(sv, srtk))

                # Sort by algorithm
                if alg == 'SURF':
                    d_surf[thresh].update({trial : df})
                elif alg == 'ORB_HAMMING':
                    d_orb[thresh].update({trial : df})
                elif alg == 'SIFT':
                    d_sift[thresh].update({trial : df})
                else:
                    pass
                
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
                    pass

    # Figure (Example)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(1,1,1)
    threshold = '500'
    trial = 'asphault-5'
    df_orb = d_orb[threshold][trial]
    df_surf = d_surf[threshold][trial]
    df_sift = d_sift[threshold][trial]
    plt.plot(df_orb['srtk'])
    plt.plot(df_orb['sv'])
    plt.plot(df_surf['sv'])
    plt.plot(df_sift['sv'])
    plt.ylim([1, 5])
    fig.show()

    ## Figure (By Surface)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(3,2,1)
    for alg, trials in d_gravel.iteritems(): # Gravel
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Gravel')
    fig.add_subplot(3,2,2)
    for alg, trials in d_asphault.iteritems(): # Asphault
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Asphault')
    fig.add_subplot(3,2,3)
    for alg, trials in d_grass.iteritems(): # Grass
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Grass')
    fig.add_subplot(3,2,4)
    for alg, trials in d_residue.iteritems(): # Residue
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Residue')
    fig.add_subplot(3,2,5)
    for alg, trials in d_corn.iteritems(): # Corn
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Corn')
    fig.add_subplot(3,2,6)
    for alg, trials in d_hay.iteritems(): # hay
        composite = pd.concat(trials)
        plt.scatter(composite['srtk'], composite['sv'], s=2, c=COLORS[alg], edgecolors='none')
    plt.axis([1, 5, 1, 5])
    plt.title('Hay')
    fig.show()

    ## Figure (By Surface)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    fig.add_subplot(3,2,1)
    for alg, trials in d_gravel.iteritems(): # Gravel
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

    # Figure (by Feature-Detector): Scatter of RTK vs Corrected Error 
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            plt.scatter(df['srtk'], df['srtk'] - df['sv'], c='r', s=1, edgecolors='none')
        plt.axis([1, 5, -0.5, 0.5])
        plt.title('SURF')
    for t in ORB_TREATMENTS:
        fig.add_subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            plt.scatter(df['srtk'], df['srtk'] - df['sv'], c='g', s=1, edgecolors='none')
        plt.axis([1, 5, -0.5, 0.5])
        plt.title('ORB')
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            plt.scatter(df['srtk'], df['srtk'] - df['sv'], c='b', s=1, edgecolors='none')
        plt.axis([1, 5, -0.5, 0.5])
        plt.title('SIFT')
    fig.show()
    
    # Figure (by Feature-Detector): RTK-groups vs. RMSE
    df_rmse = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='r')
        plt.axis([1, 5, 0, 1])
        plt.title('SURF')
    for t in ORB_TREATMENTS:          
        fig.add_subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='g')
        plt.axis([1, 5, 0, 1])
        plt.title('ORB')
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            RMSE, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='b')
        plt.axis([1, 5, 0, 1])
        plt.title('SIFT')
    fig.show()
    df_rmse.to_csv("rtk_groups-vs-rmse.csv")
    
    # Figure (by Feature-Detector): RTK-groups vs. 95th
    df_p95 = pd.DataFrame()
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for t in SURF_TREATMENTS:
        fig.add_subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            RMSE, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='r')
        plt.axis([1, 5, 0, 5])
        plt.title('SURF')
    for t in ORB_TREATMENTS:          
        fig.add_subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            RMSE, groups = p95_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='g')
        plt.axis([1, 5, 0, 5])
        plt.title('ORB')
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            RMSE, groups = rmse_by_group(df['sv'], df['srtk'], NUM_GROUPS, SPEED_RANGE)
            plt.plot(groups, RMSE, c='b')
        plt.axis([1, 5, 0, 5])
        plt.title('SIFT')
    fig.show()
    df_p95.to_csv("rtk_groups-vs-p95.csv")
    
    # Figure (by Feature-Detector): RTK vs Hz
    fig = plt.figure()
    fig.patch.set_facecolor('white')  
    for t in SURF_TREATMENTS:
        fig.add_subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='r', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            plt.scatter(df['rtk'], df['hz'], c='g', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('ORB')
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
           plt.scatter(df['rtk'], df['hz'], c='b', s=1, edgecolors='none')
        plt.axis([0, 20, 0, 50])
        plt.title('SIFT')
    fig.show()

    # Figure (by Feature-Detector): Hz vs N-features
    fig = plt.figure()
    fig.patch.set_facecolor('white')  
    for t in SURF_TREATMENTS:
        fig.add_subplot(1,1,1)
        for trial, df in d_surf[t].iteritems():
           plt.scatter(df['m'], df['hz'], c='r', s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
        plt.title('SURF')
    for t in ORB_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_orb[t].iteritems():
            plt.scatter(df['m'], df['hz'], c='g', s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
        plt.title('ORB')
    for t in SIFT_TREATMENTS:        
        fig.add_subplot(1,1,1)
        for trial, df in d_sift[t].iteritems():
           plt.scatter(df['m'], df['hz'], c='b', s=1, edgecolors='none')
        plt.axis([0, 2000, 0, 100])
        plt.title('SIFT')
    fig.show()

    # Plot all and wait...
    plt.waitforbuttonpress()
