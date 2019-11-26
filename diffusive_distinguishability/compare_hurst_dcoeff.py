import pandas as pd
import numpy as np
import ndim_homogeneous_distinguishability as hd


def hurst_dcoeff(n_dim=2, n_steps=30, hurst=0.2, dt=1, loc_std=0):
    df = hd.simulate_fbm_df(n_dim = n_dim, hurst = hurst, n_steps = n_steps, dt = dt, loc_std = loc_std)
    posterior, alpha, beta = hd.estimate_diffusion(n_dim = n_dim, dt = dt, dr = df['dr_obs'])
    mean = beta/(alpha-1)
    return mean


def scan_parameters(n_dim = [1,2,3], n_steps = [5, 10, 50, 100, 500], hurst_list = np.linspace(0.1,0.8,50), n_reps = 10):
    results = list()
    for dim in n_dim:
        for steps in n_steps:
            for hurst in hurst_list:
                replicates = [hurst_dcoeff(dim, steps, hurst) for i in range(n_reps)]
                rep_df = pd.DataFrame({'n_dim':[n_dim], 'n_steps':[n_steps], 'hurst': [hurst], 'd_coeff': [np.mean(replicates)]})
                results.append(rep_df)
    df = pd.concat(results)
    df.to_pickle('hurst_dcoeff_results')


def plot_hurst_dcoeff(df, cut=None):
    if cut is None:
        df = results
    else:
        df = results[results['hurst'] < cut]
    palette = dict(zip(df.n_steps.unique(), sb.color_palette("rocket_r", 5)))
    sb.relplot(x="hurst", y="mean",
            hue="n_steps", size="n_dim", palette=palette,
            height=8, aspect=1, facet_kws=dict(sharex=False),
            kind="line", legend="full", data=df)