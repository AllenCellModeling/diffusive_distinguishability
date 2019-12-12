import numpy as np
import pandas as pd

# import bayesian estimation package
import ndim_homogeneous_distinguishability as hd
import fbm_analysis as fa
from fbm import FBM

d_const = 0.1
n_steps = 10**3
dt = 1
n_reps = 10**4
n_dim = [1, 2, 3]
hurst = 0.75/2


n_steps_list = [10, 15, 20, 25, 35, 50, 100, 150, 200]
loc_std = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]

df1m, df2m, df3m = hd.error_sensitivity(d_const=d_const, n_steps_list=n_steps_list, dt=1, n_reps=10**4, loc_std=loc_std, mag=False, hurst=hurst)
df1m.to_pickle('saved_data/posterior_error/DE-1/DE-1_d1_mag_hurst_'+str(hurst)+'_2nm.pickle')
df2m.to_pickle('saved_data/posterior_error/DE-1/DE-1_d2_mag_hurst_'+str(hurst)+'_2nm.pickle')
df3m.to_pickle('saved_data/posterior_error/DE-1/DE-1_d3_mag_hurst_'+str(hurst)+'_2nm.pickle')
