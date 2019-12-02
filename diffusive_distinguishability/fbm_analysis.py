import ndim_homogeneous_distinguishability as hd
from fbm import FBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_fbm_df(d_const, n_dim, n_steps, dt, loc_std=0, hurst = 0.5):
    """Simulate and output a single trajectory of fractional brownian motion in a specified number of dimensions.

    :param d_const: diffusion constant in um2/s
    :param n_dim: number of spatial dimensions for simulation (1, 2, or 3)
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :param hurst: Hurst index in range (0,1), hurst=0.5 gives brownian motion
    :return: trajectory dataframe (position in n_dim dimensions, at each timepoint)
    """

    f = FBM(n = n_steps, hurst=hurst, length = n_steps*dt, method='hosking')
    t_values = f.times()
    
    fbm_sim = []
    for dim in range(n_dim):
        fbm_sim.append(f.fbm()*np.sqrt(2*d_const))

    # initialize position at origin
    df = pd.DataFrame()
 
    # at each time-step, stochastically select step size in each dimension to find new location to add to trajectory
    for i in range(n_steps):

        if i == 0:
            x_curr = [fbm_sim[dim][i] for dim in range(n_dim)]
            noise_curr = [loc_std*np.random.randn() for _dim in range(n_dim)]
            x_obs_curr = [x_curr[dim] + noise_curr[dim] for dim in range(n_dim)]
        else:
            x_curr = x_next
            x_obs_curr = x_obs_next

        x_next = [fbm_sim[dim][i+1] for dim in range(n_dim)]

        noise_next = [loc_std*np.random.randn() for _dim in range(n_dim)]
        x_obs_next = [x_next[dim] + noise_next[dim] for dim in range(n_dim)]

        dx_obs = [x_obs_next[dim] - x_obs_curr[dim] for dim in range(n_dim)]
        dx = [x_next[dim] - x_curr[dim] for dim in range(n_dim)]
        dr_obs = np.linalg.norm(dx_obs)
        dr = np.linalg.norm(dx)
        
        data = {'t_step': t_values[i], 'x': x_curr, 'x_obs': x_obs_curr, 'dx': dx, 'dx_obs': dx_obs, 'dr': dr, 'dr_obs': dr_obs}
        df = df.append(data, ignore_index=True)
        
    return df


def plot_trajectory(df):

    n_steps = len(df['x'])
    n_dim = len(df['x'].iloc[0])

    x = [df['x'].iloc[i][0] for i in range(n_steps)]
    if n_dim == 1:
        plt.plot(df['t_step'], x)
    elif n_dim == 2:
        y =  [df['x'].iloc[i][1] for i in range(n_steps)]
        plt.plot(x, y, color = 'b')
    
    if df['x_obs'].iloc[0][0] != 0:
        xx = [df['x_obs'].iloc[i][0] for i in range(n_steps)]
        if n_dim == 1:
            plt.plot(df['t_step'], xx)
        elif n_dim == 2:
            yy =  [df['x_obs'].iloc[i][1] for i in range(n_steps)]
            plt.plot(xx, yy, color = 'o')


def get_fBm_diffusivity(df, tau):
    """Get effective diffusion constant for a given timescale from fBm
    :param df: dataframe
    :param tau: timescale for diffusivity measurement
    :return: diffusivity(tau)
    """
    n_dim= len(df['x'].iloc[0])
    return np.mean(df['dr_obs']**2)/(2*n_dim*tau)


