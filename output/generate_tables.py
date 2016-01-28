import pandas as pd
import glob
import os

# Column headers (must match the .csv headers!)
columns = ['rtk','v','t','m', 'p', 'n']

# Get Subdirectories (of the form ALGORITHM-THRESHOLD)
dirs = [tuple(d.split('-')) for d in os.listdir(".") if os.path.isdir(d)]

#d = dict(zip(columns, vals))
for alg, thresh in dirs:
    print alg, thresh
    files = glob.glob(os.path.join(d,'*.csv'))
    for f in files:
        print df = pd.DataFrame.from_csv(f)
        #df_camera = df_camera.append(d, ignore_index=True)
        #pass
