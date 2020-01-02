from fbm import FBM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simulate_fbm_df(d_const, n_dim, n_steps, dt, loc_std=0, hurst=0.5):
    """Simulate and output a single trajectory of fractional brownian motion in a specified number of dimensions.

    :param d_const: diffusion constant in um2/s
    :param n_dim: number of spatial dimensions for simulation (1, 2, or 3)
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :param hurst: Hurst index in range (0,1), hurst=0.5 gives brownian motion
    :return: trajectory dataframe (position in n_dim dimensions, at each timepoint)
    """

    np.random.seed()

    # create fractional brownian motion trajectory generator
    # Package ref: https://github.com/crflynn/fbm
    f = FBM(n=n_steps, hurst=hurst, length=n_steps*dt, method='daviesharte')

    # get time list and trajectory where each timestep has and n-dimensional vector step size
    t_values = f.times()
    fbm_sim = []
    for dim in range(n_dim):
        fbm_sim.append(f.fbm()*np.sqrt(2*d_const))

    df = pd.DataFrame()
    for i in range(n_steps):

        x_curr = [fbm_sim[dim][i] for dim in range(n_dim)]
        # for initial time point, start at origin and optionally add noise
        if i == 0:
            x_obs_curr = [x_curr[dim] + loc_std*np.random.randn() for dim in range(n_dim)]
        # for latter timepoints, set "current" position to position determined by displacement out of the last timepoint
        else:
            x_obs_curr = x_obs_next

        # Get next n-dimensional position
        x_next = [fbm_sim[dim][i+1] for dim in range(n_dim)]
        # Get noise to add to next position, to get the observed position
        noise_next = [loc_std*np.random.randn() for _dim in range(n_dim)]
        x_obs_next = [x_next[dim] + noise_next[dim] for dim in range(n_dim)]
        # break current and next position into vector and magnitdue displacements
        dx_obs = [x_obs_next[dim] - x_obs_curr[dim] for dim in range(n_dim)]
        dx = [x_next[dim] - x_curr[dim] for dim in range(n_dim)]
        dr_obs = np.linalg.norm(dx_obs)
        dr = np.linalg.norm(dx)
        t = t_values[i]

        # Add timestep data to dataframe
        data = {'t_step': t, 'x': x_curr, 'x_obs': x_obs_curr, 'dx': dx, 'dx_obs': dx_obs, 'dr': dr, 'dr_obs': dr_obs}
        df = df.append(data, ignore_index=True)

    return df


def plot_trajectory(df):
    """Plot a trajectory in 1D vs time or in 2D
    :param df: trajectory dataframe formatted as in 'simulate_fbm_df' function
    """

    # Get number of steps in trajectory and number of spatial dimensions
    n_steps = len(df['x'])
    n_dim = len(df['x'].iloc[0])

    # Plot x(t) if 1D and y(x) in 2D
    x = [df['x'].iloc[i][0] for i in range(n_steps)]
    if n_dim == 1:
        plt.plot(df['t_step'], x)
    elif n_dim == 2:
        y = [df['x'].iloc[i][1] for i in range(n_steps)]
        plt.plot(x, y, color='b')

    # If noise added, plot observed trajectory
    if df['x_obs'].iloc[0][0] != 0:
        xx = [df['x_obs'].iloc[i][0] for i in range(n_steps)]
        if n_dim == 1:
            plt.plot(df['t_step'], xx)
        elif n_dim == 2:
            yy = [df['x_obs'].iloc[i][1] for i in range(n_steps)]
            plt.plot(xx, yy, color='o')


def get_fBm_diffusivity(df, tau):
    """Get effective diffusion constant for a given timescale (tau) from fBm
    :param df: dataframe
    :param tau: timescale for diffusivity measurement
    :return: diffusivity(tau)
    """
    n_dim = len(df['x'].iloc[0])
    return np.mean(df['dr_obs']**2)/(2*n_dim*tau)
