import numpy as np
import math
import scipy.stats
import scipy.special
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import multiprocessing as mp


# Simulation of trajectories and storage of trajectory data

def simulate_diffusion_df(n_dim, D, T, dt, loc_std=0):
    """Simulate and output a single trajectory of homogeneous diffusion in a specified number of dimensions.
    :param n_dim: number of spatial dimensions for simulation (1, 2, or 3)
    :param D: diffusion constant (um2/s)
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return: trajectory dataframe (position in n_dim dimensions, at each timepoint)
    """

    np.random.seed()

    # initialize position at origin
    x0 = np.zeros(n_dim)
    x = x0
    x_obs = [sum(i) for i in zip(x, [loc_std*np.random.randn()for dim in range(n_dim)])] 
    df = pd.DataFrame()
    
    # at each time-step, stochastically select step size in each dimension to find new location to add to trajectory
    for t in np.linspace(0, (T-1)*dt, T):
        
        dx = [np.sqrt(2*D*dt)*np.random.randn() for _dim in range(n_dim)]
        noise = [loc_std*np.random.randn() for _dim in range(n_dim)]
        
        x_new = [sum(i) for i in zip(x, dx)]
        x_obs_new = [sum(i) for i in zip(x_new, noise)]
        dx_obs = [sum(i) for i in zip(x_obs_new, np.negative(x_obs))]
        dr = np.linalg.norm(dx)
        dr_obs = np.linalg.norm(dx_obs)
        
        data = {'tstep': t, 'x': x, 'x_obs': x_obs, 'dx': dx, 'dx_obs': dx_obs, 'dr': dr, 'dr_obs': dr_obs}
        df = df.append(data, ignore_index=True)
        x = x_new
        x_obs = x_obs_new
        
    return df


def trajectory_df_from_data(trajectory):
    """If you are using experimental rather than simulated trajectories:
    this will intake a timelapse trajectory and generate a dataframe compatible with this notebook for analysis.
    :param trajectory: list or array of spatial positions, where each entry is the position at a single timepoint
    (may be 1D, 2D or 3D)
    :return: dataframe containing trajectory, n-dimensional displacement vectors for each timestep, and step size
    magnitudes for each timestep
    """

    df = pd.DataFrame()

    for t in range(len(trajectory)-1):
        dx_obs = trajectory[t+1] - trajectory[t]
        data = {'tstep': t, 'x_obs': trajectory[t], 'dx_obs': dx_obs, 'dr_obs': np.linalg.norm(dx_obs)}
        df = df.append(data, ignore_index=True)
        
    return df


# Stats utils - posterior generation and comparison

def estimate_diffusion(n_dim, dt, dr, prior=scipy.stats.distributions.invgamma(0, scale=0)):
    """Returns the posterior estimate for the diffusion constant given the displacement data and the prior.
    :param n_dim: number of spatial dimensions for simulation (1, 2, or 3)
    :param dt: timestep size (s)
    :param dr: list of normed step sizes from a single trajectory (um)
    :param prior: inverse gamma prior distribution estimate for the diffusion constant
    :return: inverse gamma posterior distribution estimate for the diffusion constant
    """
    # get step sizes and calculate alpha and beta from them to characterize a inverse gamma distribution
    dr = np.array(dr)/np.sqrt(2*n_dim*dt)
    alpha0, beta0 = invgamma_fullparams(prior)
    alpha,  beta = len(dr), sum(dr**2)
        
    return scipy.stats.distributions.invgamma(alpha0+alpha, scale=beta0+beta), alpha0+alpha, beta0+beta

    
def generate_posterior(n_dim, D, T, dt, loc_std=0):
    """
    Simulate a single trajectory and find the diffusion constant posterior (inverse gamma) distribution.
    :param n_dim: number of spatial dimensions
    :param D: diffusion constant (um2/s)
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return alpha, beta: scale and shape parameters for inverse gamma posterior for a diffusive trajectory
    """
    
    # get dataframe of (x, dx) for a trajectory of length T in homogeneous diffusion constant D
    df = simulate_diffusion_df(n_dim, D, T, dt, loc_std)
    
    # estimate posterior D distribution using prior/posterior with inverse gamma form
    prior = scipy.stats.distributions.invgamma(0, scale=0)
    posterior, alpha, beta = estimate_diffusion(n_dim=n_dim, dt=dt, dr=df['dr_obs'], prior=prior)

    # return df, posterior, alpha, beta
    return alpha, beta


def get_posterior_set(n_dim, D, T, dt, N, loc_std=0):
    """
    Repeat analysis generating a posterior D distribution per trajectory for multiple trajectories and
    return (1) full set and (2) median values of distribution fit parameters.
    :param n_dim: number of spatial dimensions
    :param D: diffusion constant (um2/s)
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param N: number of trajectories
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return alpha, beta, alpha_std, beta_std, alphas, betas: medians, std devs and arrays of scale and 
    shape parameters for inverse gamma posteriors for N diffusive trajectories
    """
    
    n_cpus = mp.cpu_count()
    with mp.Pool(n_cpus-2) as pool:
        results = pool.starmap(generate_posterior, [(n_dim, D, T, dt, loc_std) for _n in range(N)])
    alphas = [result[0] for result in results]
    betas = [result[1] for result in results]
    
    # calculate median values for alpha and beta from N simulations
    alpha = np.nanmedian(np.asarray(alphas))
    beta = np.nanmedian(np.asarray(betas))
    
    return alpha, beta, alphas, betas


def invgamma_fullparams(dist):
    """Return the alpha,beta parameterization of the inverse gamma distirbution
    :param dist: scipy inverse gamma distribution
    :return: alpha and beta parameters characterizing this inverse gamma distribution"""

    return dist.args[0], dist.kwds['scale']


def invgamma_kldiv(param1, param2):
    """Compute KL divergence of two inverse gamma distributions (ref: https://arxiv.org/pdf/1605.01019.pdf)
    :param param1: list containing alpha and beta parameters characterizing inverse gamma distribution 1
    :param param2: list containing alpha and beta parameters characterizing inverse gamma distribution 2
    :return: KL divergence of two inverse gamma distributions
    """

    # unpack distribution parameters
    alpha1 = param1[0]
    beta1 = param1[1]
    alpha2 = param2[0]
    beta2 = param2[1]

    term1 = (alpha1 - alpha2)*scipy.special.digamma(alpha1)
    term2 = (beta2*alpha1/beta1)
    term3 = -alpha1
    term4a = (alpha2+1)*np.log(beta1)
    term4b = math.lgamma(alpha2)
    term4c = -np.log(beta1)
    term4d = -alpha2*np.log(beta2)
    term4e = -math.lgamma(alpha1)
    
    return term1 + term2 + term3 + term4a + term4b + term4c + term4d + term4e


# Visualization and analyses

def compare2(n_dim, D1, mult, T, dt, N, loc_std=0):
    """
    For one pair of diffusion constants (D, D*mult) get KL divergence of their posteriors, where the posteriors are
    generated from an alpha and beta which are the median values from repeating posterior estimation N times
    :param n_dim: number of spatial dimensions
    :param D1: diffusion constant (um2/s)
    :param mult: multiplier to get D2 = mult*D1
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param N: number of trajectories
    :param loc_std: standard deviation of localization error (um):param D1: diffusion constant
    """

    D2 = D1*mult

    # for N trajectories of length T, get median values for posterior parameters and their std
    alpha1, beta1, alpahs1, betas1 = get_posterior_set(n_dim, D1, T, dt, N, loc_std)
    alpha2, beta2, alphas2, betas2 = get_posterior_set(n_dim, D2, T, dt, N, loc_std)
    
    # plot both posteriors
    xx = np.linspace(0, 1.5*D2, 50)
    plt.plot(xx, scipy.stats.distributions.invgamma(alpha1, scale=beta1).pdf(xx), label='D1 posterior')
    plt.plot(xx, scipy.stats.distributions.invgamma(alpha2, scale=beta2).pdf(xx), label='D2 posterior')
    plt.axvline(x=D1, linestyle=':', color='blue')
    plt.axvline(x=D2, linestyle=':', color='orange')
    plt.legend()
    plt.xlabel('Diffusion Constant')
    plt.ylabel('Probability density')
    
    # print posterior-pair KL divergence and its inverse
    KL = invgamma_kldiv([alpha1, beta1], [alpha2, beta2])
    print('KL div: ' + str(KL))
    print('Inverse: ' + str(1./KL))

    
def fill_heatmap_gen(n_dim, D, mult_list, T, dt, N, loc_std=0):
    """
    Generate a heatmap of KL divergence values for pairwise comparison of diffusion constant posterior
    distributions. Compared posteriors are generated by scanning through pairings of [D, mult*D] where mult takes on 
    the range of values provided by mult_list and trajectory lengths. For each pair of diffusion constants, 
    generate a trajectory of length T and find the associated posterior parameter fit, repeating N times to get 
    median parameter values (alpha, beta). Use these median values of alpha and beta to select one posterior D
    distribution for that diffusion constant. Repeat this process for diffusion constant D*mult, then calculate the
    KL divergence of the posteriors for (D, D*multiplier) and store in dataframe. Repeat for all pairs of
    (T, multiplier) to fill the dataframe. The results is a heatmap of how distinguishable two diffusion constants are,
    conditional upon their relative values and the length of trajectories used.
    
    :param n_dim: number of spatial dimensions
    :param D: diffusion constant (um2/s)
    :param mult_list: list of multipliers to get set of D2 values, where D2 = mult*D
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param N: number of trajectories
    :param loc_std: standard deviation of localization error (um):param D1: diffusion constant
    :return df: dataframe containing the pairwise KL divergences
    """
    
    # find input parameter that is a list, indicating that it is the parameter to be swept over
    params = [D, T, dt, loc_std]
    is_list = [isinstance(item,(list,np.ndarray)) for item in params]
    param_ind = int(np.where(is_list)[0])

    # make sure only only parameter (besides mult_list) has been entered as a list to be the second axis of sweep figure
    if sum(is_list)!=1:
        raise ValueError('Only one input other than "mult" may be multivalued.')

    # set sweep parameter as x axis and create dataframe
    x_list = params[param_ind]
    df = pd.DataFrame(columns=x_list, index=mult_list)

    # loop through D pairs and find their posterior fit parameters and pairwise KL divergences
    for x_ind in range(len(x_list)):
        for m_ind in range(len(mult_list)):
            
            # for individual model run, get single sweep parameter value and select pair of diffusion constants
            params[param_ind] = x_list[x_ind]
            D1 = params[0]
            D2 = D1*mult_list[m_ind]
            T = params[1]
            dt = params[2]
            loc_std = params[3]
            
            # calculate posterior fit params and their std's
            alpha1_med, beta1_med, alphas1, betas1 = get_posterior_set(n_dim, D1, T, dt, N, loc_std)
            alpha2_med, beta2_med, alphas2, betas2 = get_posterior_set(n_dim, D2, T, dt, N, loc_std)
            
            # store KL divergence of posteriors in dataframe
            df.iat[m_ind,x_ind] = invgamma_kldiv([alpha1_med, beta1_med], [alpha2_med, beta2_med])

    return df[df.columns].astype(float)


# Error visualization analysis


def show_error_hist(n_dim, p_error):
    """
    Plot figure with 3 subplots, where each subplot is a histogram of the percent errors from all runs in a given number
     of spatial dimensions
    :param n_dim: number of spatial dimensions
    :param p_error: array of percent error for all runs in each number of spatial dimensions
    """
    fig, axs = plt.subplots(1, len(n_dim), figsize=(18,6))
        
    for i in range(len(n_dim)):
        print('Dim = '+str(i+1)+': '+str(np.mean(p_error[i]))+' +- '+str(np.std(p_error[i])/len(p_error[i])))
        sns.distplot(p_error[i], bins=30, ax=axs[i])
        axs[i].set_xlabel('% estimation error')
        axs[i].set_ylabel('Count')
        axs[i].set_title('Dim = '+str(i+1))
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
    
def get_single_error(args):
    """
    Unpack args to generate single posterior, and calculate percent error of posterior mean relative to the true value
    :param args: set of parameters listed below as args, all packaged as one object for the sake of parallel processing
    :arg dim: number of spatial dimensions
    :arg D: diffusion constant (um2/s) whose estimator error we want to calculate
    :arg T: trajectory length (number of steps)
    :arg dt: timestep size (s)
    :arg n: trajectory number
    :arg loc_std: standard deviation of localization error (um)
    :return: percent error for a single posterior mean relative to true value
    """
    dim, D, T, dt, n, loc_std = args
    alpha, beta = generate_posterior(dim, D, T, dt, loc_std)
    post_mean = scipy.stats.distributions.invgamma(alpha, scale=beta).mean()
    return 100*((post_mean - D)/D)
    
    
def get_dim_error(n_dim, D, T, dt, N, show_plot, loc_std=0):
    """
    Given a diffusion constant, get the posterior for a trajectory of length T and timestep dt. Repeat N times
    and report the percent error of the mean posterior value vs true D value and plot a histogram of these values.
    :param n_dim: number of spatial dimensions
    :param D: diffusion constant (um2/s) whose estimator error we want to calculate
    :param T: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param N: number of trajectories
    :param show_plot: T/F flag of whether or not to display histograms of estimator errors
    :param loc_std: standard deviation of localization error (um)
    :return p_error: array of percent error between mean posterior estimation and true value for each run with each
    number of dimensions
    """

    n_cpus = mp.cpu_count()
    p_error = []
    for dim in n_dim:
        input_args = []
        for n in range(N):
            input_args.append((dim, D, T, dt, n, loc_std))   
        with mp.Pool(n_cpus-2) as pool:
            results = pool.map(get_single_error, input_args)
        p_error.append(results)
        
    # uncomment below to save error results as .npy
    # np.save('std_'+str(loc_std), p_error)
    
    # display histogram of percent errors for N runs of simulation with diffusion constant D
    if show_plot:
        show_error_hist(n_dim, p_error)

    return p_error


def error_sensitivity(D, T_list, dt, N, loc_std):
    """
    Look at how the mean and median percent error of the posterior mean relative to the true value
    depend on the trajectory length used to generate posteriors and number of reps we run
    :param D: diffusion constants (um2/s)
    :param T_list: list of trajectory lengths to test
    :param dt: timestep(s) used to generate trajectories (s)
    :param N: number(s) of reps to run to calculate mean and mediate percent error
    :param loc_std: standard deviation for guassian localization error (um)
    :return: three dataframes (for 1, 2, and 3 dimensions); each contains the mean percent posterior error relative to true diffusion
    constant value, for all pairs of trajectory lengths and localization errors includes in these two input lists
    """
    
    size = 8
    plt.subplots(1,3, figsize=(3*size, size))
    
    data1 = np.zeros((len(T_list), len(loc_std)))
    data2 = np.zeros((len(T_list), len(loc_std)))
    data3 = np.zeros((len(T_list), len(loc_std)))
    for T in T_list:
        for std in loc_std:
            p_error = get_dim_error([1, 2, 3], D, T, dt, N, False, std)
            Ti, Ei = np.asarray(T_list).searchsorted(T), np.asarray(loc_std).searchsorted(std)
            data1[Ti, Ei] = np.mean(p_error[0])
            data2[Ti, Ei] = np.mean(p_error[1])
            data3[Ti, Ei] = np.mean(p_error[2])
    df1 = pd.DataFrame(data=data1, columns=loc_std, index=T_list)
    df2 = pd.DataFrame(data=data2, columns=loc_std, index=T_list)
    df3 = pd.DataFrame(data=data3, columns=loc_std, index=T_list)
    
    return df1, df2, df3

    
# Plotting support functions

def get_ticks(tick_vals, n_round, n_ticks):
    """
    Round tick values and keep only some ticks to improve readability
    :param tick_vals: tick values
    :param n_round: number of decimal places to round to
    :param n_ticks: number of ticks to keep
    :return ticks: list of axis tick values to display
    """
    ticks = tick_vals.round(n_round)
    
    keep_ticks = ticks[::int(len(ticks)/n_ticks)]
    ticks = ['' for t in ticks]
    ticks[::int(len(ticks)/n_ticks)] = keep_ticks
    
    return ticks


def plot_df_results(df1, df2, n_round, n_ticks, size, title1, title2, x_lab, y_lab):
    """
    Plot two df heatmaps as two subplots of one figure. They share x and y axis labels but have differing titles
    :param df1: df to visualize
    :param df2: second df to visualize (often log of df1)
    :param n_round: number of axis tick decimal places to round to
    :param n_ticks: number of axis ticks to keep
    :param size: figure size
    :param title1: plot title for left (df1) panel
    :param title2: plot title for left (df2) panel
    :param x_lab: x axis label
    :param y_lab: y axis label
    """
    # set y ticks: round and only display some to improve readability
    y_ticks = get_ticks(df1.index.values, n_round, n_ticks)
    fig, axs = plt.subplots(1, 2, figsize=(2*size, size))
    
    sns.heatmap(df1, yticklabels=y_ticks, cbar_kws={'label': title1}, ax=axs[0], cmap='viridis')
    axs[0].set(xlabel=x_lab, ylabel=y_lab, title=title1)
    axs[0].invert_yaxis()
    
    sns.heatmap(df2, yticklabels=y_ticks, cbar_kws={'label': title2}, ax=axs[1], cmap='viridis')
    axs[1].set(xlabel=x_lab, ylabel=y_lab, title=title2)
    axs[1].invert_yaxis()
