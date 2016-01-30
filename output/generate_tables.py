import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def histogram(x, res=0.1, minima=0, maxima=20):
    hist, bins = np.histogram(x, int(maxima / res), [minima, maxima])
    return hist, bins
def rmse(predictions, targets):
    return np.sqrt(((np.array(predictions) - np.array(targets)) ** 2).mean())
def rmse_by_group(y, x, bins, lims):
    groups_a = np.linspace(lims[0], lims[1], bins)
    groups_b = np.linspace(lims[0], lims[1], bins) + (lims[1] - lims[0] / bins)
    groups = zip(groups_a, groups_b)
    return [rmse(y[np.logical_and(y>a, y<b)], x[np.logical_and(y>a, y<b)]) for (a,b) in groups], groups_a
def normalize(x):
    norm=np.linalg.norm(x)
    if norm==0: 
       return x
    return x/norm

# Iterate through all trials
if __name__ == '__main__':
    
    PLOT_TREATMENTS = ['500']
    PLOT_SURFACES = ['residue']
    
    # Column headers (must match the .csv headers!)
    columns = ['rtk','v','t','m', 'p', 'n']

    # Get Subdirectories (of the form ALGORITHM-THRESHOLD)
    dirs = [d for d in os.listdir(".") if os.path.isdir(d)]
    treatments = [tuple(d.split('-')) for d in os.listdir(".") if os.path.isdir(d)]
    trials = [f.split('/')[-1].split('.')[0] for f in glob.glob("../data/*.csv")]
    
    # Make dictionaries to hold DataFrames by algorithm
    d_surf = { t:{} for a,t in treatments}
    d_orb = { t:{} for a,t in treatments}
    d_sift = { t:{} for a,t in treatments}
    
    for d in dirs:
        alg, thresh = d.split('-')
        files = glob.glob(os.path.join(d,'*.csv'))
        for f in files:
            print f
            df = pd.DataFrame.from_csv(f)
            trial = f.split('/')[-1].split('.')[0]
            e = df['rtk'] - df['v']
            h, b = np.histogram(e, (20/0.1), [-10, 10])
            df_h = pd.DataFrame({'h':h})
            df_b = pd.DataFrame({'b':b})
            df = df.join(df_h)
            df_plus = df.join(df_b)
            if alg == 'SURF':
                d_surf[thresh].update({trial : df_plus})
            elif alg == 'ORB':
                d_orb[thresh].update({trial : df_plus})
            elif alg == 'SIFT':
                d_sift[thresh].update({trial : df_plus})

    # Print Figures
    for t in PLOT_TREATMENTS:

        plt.subplot(3,1,1)
        h = np.zeros(199)
        for trial, df in d_surf[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                h = df['h'][~np.isnan(df['h'])]
                b = df['b'][~np.isnan(df['b'])][:-1]
                plt.plot(b, normalize(h), c='r')
        plt.plot(b, normalize(h), c='r')
        plt.axis([-1, 1, 0, 1])

        plt.subplot(3,1,2)
        h = np.zeros(199)
        for trial, df in d_orb[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                h = df['h'][~np.isnan(df['h'])]
                b = df['b'][~np.isnan(df['b'])][:-1]
                plt.plot(b, normalize(h), c='g')
        plt.axis([-1, 1, 0, 1])
        plt.title('ORB')

        plt.subplot(3,1,3) 
        h = np.zeros(199)
        for trial, df in d_sift[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                h = df['h'][~np.isnan(df['h'])]
                b = df['b'][~np.isnan(df['b'])][:-1]
                plt.plot(b, normalize(h), c='b')
        plt.plot(b, normalize(h), c='b')
        plt.axis([-1, 1, 0, 1])
        plt.title('SIFT')
    plt.show()

    # Print Figures
    for t in PLOT_TREATMENTS:
        
        plt.subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['v'], df['rtk'] - df['v'], c='r', s=1)
        plt.axis([0, 20, -5, 5])
        plt.title('SURF')

        plt.subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['v'], df['rtk'] - df['v'], c='g', s=1)
        plt.axis([0, 20, -5, 5])
        plt.title('ORB')

        plt.subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['v'], df['rtk'] - df['v'], c='b', s=1)
        plt.axis([0, 20, -5, 5])
        plt.title('SIFT')
        
    plt.show()

    # Print Figures    
    for t in PLOT_TREATMENTS:
        
        plt.subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                RMSE, groups = rmse_by_group(df['v'], df['rtk'], 20, [0,20])
                plt.plot(groups, RMSE, c='r')
        plt.axis([0, 20, 0, 20])
        plt.title('SURF')
        
        plt.subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                RMSE, groups = rmse_by_group(df['v'], df['rtk'], 20, [0,20])
                plt.plot(groups, RMSE, c='g')
        plt.axis([0, 20, 0, 20])
        plt.title('ORB')
        
        plt.subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                RMSE, groups = rmse_by_group(df['v'], df['rtk'], 20, [0,20])
                plt.plot(groups, RMSE, c='b')
        plt.axis([0, 20, 0, 20])
        plt.title('SIFT')
    plt.show()

    # Print Figures    
    for t in PLOT_TREATMENTS:
        
        plt.subplot(3,1,1)
        for trial, df in d_surf[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['rtk'], df['hz'], c='r', s=2)
        plt.axis([0, 20, 0, 50])
        plt.title('SURF')
        
        plt.subplot(3,1,2)
        for trial, df in d_orb[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['rtk'], df['hz'], c='g', s=2)
        plt.axis([0, 20, 0, 50])
        plt.title('ORB')
        
        plt.subplot(3,1,3)
        for trial, df in d_sift[t].iteritems():
            if trial.split('-')[0] in PLOT_SURFACES or PLOT_SURFACES == []:
                plt.scatter(df['rtk'], df['hz'], c='b', s=2)
        plt.axis([0, 20, 0, 50])
        plt.title('SIFT')
    plt.show()
