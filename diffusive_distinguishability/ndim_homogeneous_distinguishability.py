import numpy as np
import math
import scipy.stats
import scipy.special
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor

# Simulation of trajectories and storage of trajectory data


def simulate_diffusion_df(n_dim, d_const, n_steps, dt, loc_std=0):
    """Simulate and output a single trajectory of homogeneous diffusion in a specified number of dimensions.

    :param n_dim: number of spatial dimensions for simulation (1, 2, or 3)
    :param d_const: diffusion constant (um2/s)
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return: trajectory dataframe (position in n_dim dimensions, at each timepoint)
    """

    np.random.seed()

    # initialize position at origin
    x0 = np.zeros(n_dim)
    x = x0
    x_obs = [sum(i) for i in zip(x, [loc_std*np.random.randn() for _dim in range(n_dim)])]
    df = pd.DataFrame()
    
    # at each time-step, stochastically select step size in each dimension to find new location to add to trajectory
    for t in np.linspace(0, (n_steps-1)*dt, n_steps):
        
        dx = [np.sqrt(2*d_const*dt)*np.random.randn() for _dim in range(n_dim)]
        noise = [loc_std*np.random.randn() for _dim in range(n_dim)]
        
        x_new = [sum(i) for i in zip(x, dx)]
        x_obs_new = [sum(i) for i in zip(x_new, noise)]
        dx_obs = [sum(i) for i in zip(x_obs_new, np.negative(x_obs))]
        dr = np.linalg.norm(dx)
        dr_obs = np.linalg.norm(dx_obs)
        
        data = {'t_step': t, 'x': x, 'x_obs': x_obs, 'dx': dx, 'dx_obs': dx_obs, 'dr': dr, 'dr_obs': dr_obs}
        df = df.append(data, ignore_index=True)
        x = x_new
        x_obs = x_obs_new
        
    return df


def trajectory_df_from_data(trajectory):
    """If you are using experimental rather than simulated trajectories:
    this is an example function for how you might import your own timelapse trajectory and put into the required
    dataframe format, compatible with this notebook for analysis. This function will likely require edits for
    individual use, to make it compatible with your input trajectory format.

    :param trajectory: list or array of spatial positions, where each entry is the position at a single timepoint
    (may be 1D, 2D or 3D)
    :return: dataframe containing trajectory, n-dimensional displacement vectors for each timestep, and step size
    magnitudes for each timestep
    """

    df = pd.DataFrame()

    for t in range(len(trajectory)-1):
        dx_obs = trajectory[t+1] - trajectory[t]
        data = {'t_step': t, 'x_obs': trajectory[t], 'dx_obs': dx_obs, 'dr_obs': np.linalg.norm(dx_obs)}
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

    
def generate_posterior(n_dim, d_const, n_steps, dt, loc_std=0):
    """
    Simulate a single trajectory and find the diffusion constant posterior (inverse gamma) distribution.

    :param n_dim: number of spatial dimensions
    :param d_const: diffusion constant (um2/s)
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return alpha, beta: scale and shape parameters for inverse gamma posterior for a diffusive trajectory
    """
    
    # get dataframe of (x, dx) for a trajectory of length n_steps in homogeneous diffusion constant d_const
    df = simulate_diffusion_df(n_dim, d_const, n_steps, dt, loc_std)
    
    # estimate posterior diffusion constant distribution using prior/posterior with inverse gamma form
    prior = scipy.stats.distributions.invgamma(0, scale=0)
    posterior, alpha, beta = estimate_diffusion(n_dim=n_dim, dt=dt, dr=df['dr_obs'], prior=prior)

    # return df, posterior, alpha, beta
    return alpha, beta


def get_posterior_set(n_dim, d_const, n_steps, dt, n_reps, loc_std=0):
    """
    Repeat analysis generating a posterior diffusion constant distribution per trajectory for multiple trajectories and
    return (1) full set and (2) median values of distribution fit parameters.

    :param n_dim: number of spatial dimensions
    :param d_const: diffusion constant (um2/s)
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param n_reps: number of trajectory replicates
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return alpha, beta, alpha_std, beta_std, alphas, betas: medians, std deviations and arrays of scale and
    shape parameters for inverse gamma posteriors for n_reps diffusive trajectories
    """
    
    with ProcessPoolExecutor() as exe:
        results = list(exe.map(generate_posterior, *zip(*((n_dim, d_const, n_steps, dt, loc_std) for _n in range(n_reps)))))
    alphas = [result[0] for result in results]
    betas = [result[1] for result in results]
    
    # calculate median values for alpha and beta from n_reps simulations
    alpha = np.nanmedian(np.asarray(alphas))
    beta = np.nanmedian(np.asarray(betas))
    
    return alpha, beta, alphas, betas


def invgamma_fullparams(dist):
    """Return the alpha,beta parameterization of the inverse gamma distribution.

    :param dist: scipy inverse gamma distribution
    :return: alpha and beta parameters characterizing this inverse gamma distribution"""

    return dist.args[0], dist.kwds['scale']


def invgamma_kldiv(param1, param2):
    """Compute KL divergence of two inverse gamma distributions (ref: https://arxiv.org/pdf/1605.01019.pdf).

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


def compare2(n_dim, d_const1, mult, n_steps, dt, n_reps, loc_std=0):
    """
    For one pair of diffusion constants (d_const, d_const*mult) get KL divergence of their posteriors, where the
    posteriors are generated from an alpha and beta which are the median values from repeating posterior estimation
    n_reps times.

    :param n_dim: number of spatial dimensions
    :param d_const1: diffusion constant (um2/s)
    :param mult: multiplier to get d_const2 = mult*d_const1
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param n_reps: number of trajectory replicates
    :param loc_std: standard deviation of localization error (um)
    """

    d_const2 = d_const1*mult

    # for n_reps trajectories of length n_steps, get median values for posterior parameters and their std
    alpha1, beta1, alphas1, betas1 = get_posterior_set(n_dim, d_const1, n_steps, dt, n_reps, loc_std)
    alpha2, beta2, alphas2, betas2 = get_posterior_set(n_dim, d_const2, n_steps, dt, n_reps, loc_std)
    
    # plot both posteriors
    xx = np.linspace(0, 1.5*d_const2, 50)
    plt.plot(xx, scipy.stats.distributions.invgamma(alpha1, scale=beta1).pdf(xx), label='D1 posterior')
    plt.plot(xx, scipy.stats.distributions.invgamma(alpha2, scale=beta2).pdf(xx), label='D2 posterior')
    plt.axvline(x=d_const1, linestyle=':', color='blue')
    plt.axvline(x=d_const2, linestyle=':', color='orange')
    plt.legend()
    plt.xlabel('Diffusion Constant')
    plt.ylabel('Probability density')
    
    # print posterior-pair KL divergence and its inverse
    kl_div = invgamma_kldiv([alpha1, beta1], [alpha2, beta2])
    print('KL div: ' + str(kl_div))
    print('Inverse: ' + str(1./kl_div))

    
def fill_heatmap_gen(n_dim, d_const, mult_list, n_steps, dt, n_reps, loc_std=0):
    """
    Generate a heatmap of KL divergence values for pairwise comparison of diffusion constant posterior
    distributions. Compared posteriors are generated by scanning through pairings of [d_const, mult*d_const] where mult
    takes on the range of values provided by mult_list and trajectory lengths. For each pair of diffusion constants,
    generate a trajectory of length n_steps and find the associated posterior parameter fit, repeating n_reps times to
    get median parameter values (alpha, beta). Use these median values of alpha and beta to select one posterior
    diffusion constant distribution for that diffusion constant. Repeat this process for diffusion constant
    d_const*mult, then calculate the KL divergence of the posteriors for (d_const, d_const*multiplier) and store in
    dataframe. Repeat for all pairs of (n_steps, multiplier) to fill the dataframe. The results is a heatmap of how
    distinguishable two diff constants are, conditional upon their relative values and the length of trajectories used.
    
    :param n_dim: number of spatial dimensions
    :param d_const: diffusion constant (um2/s)
    :param mult_list: list of multipliers to get set of d_const2 values, where d_const2 = mult*d_const
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param n_reps: number of trajectories
    :param loc_std: standard deviation of localization error (um)
    :return df: dataframe containing the pairwise KL divergences
    """
    
    # find input parameter that is a list, indicating that it is the parameter to be swept over
    params = [d_const, n_steps, dt, loc_std]
    is_list = [isinstance(item, (list, np.ndarray)) for item in params]
    param_ind = int(np.where(is_list)[0])

    # make sure only only parameter (besides mult_list) has been entered as a list to be the second axis of sweep figure
    if sum(is_list) != 1:
        raise ValueError('Only one input other than "mult" may be multivalued.')

    # set sweep parameter as x axis and create dataframe
    x_list = params[param_ind]
    df = pd.DataFrame(columns=x_list, index=mult_list)

    # loop through d_const pairs and find their posterior fit parameters and pairwise KL divergences
    for x_ind in range(len(x_list)):
        for m_ind in range(len(mult_list)):
            
            # for individual model run, get single sweep parameter value and select pair of diffusion constants
            params[param_ind] = x_list[x_ind]
            d_const1 = params[0]
            d_const2 = d_const1*mult_list[m_ind]
            n_steps = params[1]
            dt = params[2]
            loc_std = params[3]
            
            # calculate posterior fit params and their std's
            alpha1_med, beta1_med, alphas1, betas1 = get_posterior_set(n_dim, d_const1, n_steps, dt, n_reps, loc_std)
            alpha2_med, beta2_med, alphas2, betas2 = get_posterior_set(n_dim, d_const2, n_steps, dt, n_reps, loc_std)
            
            # store KL divergence of posteriors in dataframe
            df.iat[m_ind, x_ind] = invgamma_kldiv([alpha1_med, beta1_med], [alpha2_med, beta2_med])

    return df[df.columns].astype(float)


# Error visualization analysis


def show_error_hist(n_dim, p_error):
    """
    Plot figure with 3 subplots, where each subplot is a histogram of the percent errors from all runs in a given number
     of spatial dimensions.

    :param n_dim: number of spatial dimensions
    :param p_error: array of percent error for all runs in each number of spatial dimensions
    """
    fig, axs = plt.subplots(1, len(n_dim), figsize=(18, 6))
        
    for i in range(len(n_dim)):
        print('Dim = '+str(i+1)+': '+str(np.mean(p_error[i]))+' +- '+str(np.std(p_error[i])/len(p_error[i])))
        sns.distplot(p_error[i], bins=30, ax=axs[i])
        axs[i].set_xlabel('% estimation error')
        axs[i].set_ylabel('Count')
        axs[i].set_title('Dim = '+str(i+1))
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    
    
def get_single_error(dim, d_const, n_steps, dt, n, loc_std):
    """
    Generate single posterior and calculate percent error of posterior mean relative to the true value.

    :param dim: number of spatial dimensions
    :param d_const: diffusion constant (um2/s) whose estimator error we want to calculate
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param n: trajectory number
    :param loc_std: standard deviation of localization error (um)
    :return: percent error for a single posterior mean relative to true value
    """

    alpha, beta = generate_posterior(dim, d_const, n_steps, dt, loc_std)
    post_mean = scipy.stats.distributions.invgamma(alpha, scale=beta).mean()
    return 100*((post_mean - d_const)/d_const)
    
    
def get_dim_error(n_dim, d_const, n_steps, dt, n_reps, show_plot, loc_std=0):
    """
    Given a diffusion constant, get the posterior for a trajectory of length n_steps and timestep dt. Repeat n_reps
    times and report/plot hist of the percent error of the mean posterior values vs true diffusivity values.

    :param n_dim: number of spatial dimensions
    :param d_const: diffusion constant (um2/s) whose estimator error we want to calculate
    :param n_steps: trajectory length (number of steps)
    :param dt: timestep size (s)
    :param n_reps: number of trajectory replicates
    :param show_plot: T/F flag of whether or not to display histograms of estimator errors
    :param loc_std: standard deviation of localization error (um)
    :return p_error: array of percent error between mean posterior estimation and true value for each run with each
    number of dimensions
    """

    p_error = []
    for dim in n_dim:
        with ProcessPoolExecutor() as exe:
            results = list(exe.map(get_single_error, *zip(*((dim, d_const, n_steps, dt, n, loc_std) for n in range(n_reps)))))
        p_error.append(results)
        
    # uncomment below to save error results as .npy
    #np.save('std_'+str(loc_std), p_error)
    
    # display histogram of percent errors for n_reps runs of simulation with diffusion constant d_const
    if show_plot:
        show_error_hist(n_dim, p_error)

    return p_error


def error_sensitivity(d_const, n_steps_list, dt, n_reps, loc_std):
    """
    Look at how the mean and median percent error of the posterior mean relative to the true value
    depend on the trajectory length used to generate posteriors and number of reps we run.

    :param d_const: diffusion constants (um2/s)
    :param n_steps_list: list of trajectory lengths to test
    :param dt: timestep(s) used to generate trajectories (s)
    :param n_reps: number(s) of reps to run to calculate mean and mediate percent error
    :param loc_std: standard deviation for Gaussian localization error (um)
    :return: three dataframes (for 1, 2, and 3 dimensions); each contains the mean percent posterior error relative to
    true diffusion constant value, for all pairs of trajectory lengths and localization errors includes in these two
    input lists
    """
    
    size = 8
    plt.subplots(1, 3, figsize=(3*size, size))
    
    data1 = np.zeros((len(n_steps_list), len(loc_std)))
    data2 = np.zeros((len(n_steps_list), len(loc_std)))
    data3 = np.zeros((len(n_steps_list), len(loc_std)))
    for n_steps in n_steps_list:
        for std in loc_std:
            p_error = get_dim_error([1, 2, 3], d_const, n_steps, dt, n_reps, False, std)
            ind_steps, ind_error = np.asarray(n_steps_list).searchsorted(n_steps), np.asarray(loc_std).searchsorted(std)
            data1[ind_steps, ind_error] = np.mean(p_error[0])
            data2[ind_steps, ind_error] = np.mean(p_error[1])
            data3[ind_steps, ind_error] = np.mean(p_error[2])
    df1 = pd.DataFrame(data=data1, columns=loc_std, index=n_steps_list)
    df2 = pd.DataFrame(data=data2, columns=loc_std, index=n_steps_list)
    df3 = pd.DataFrame(data=data3, columns=loc_std, index=n_steps_list)
    
    return df1, df2, df3

    
# Plotting support functions


def get_ticks(tick_values, n_round, n_ticks):
    """
    Round tick values and keep only some ticks to improve readability.

    :param tick_values: tick values
    :param n_round: number of decimal places to round to
    :param n_ticks: number of ticks to keep
    :return ticks: list of axis tick values to display
    """
    ticks = tick_values.round(n_round)
    
    keep_ticks = ticks[::int(len(ticks)/n_ticks)]
    ticks = ['' for _t in ticks]
    ticks[::int(len(ticks)/n_ticks)] = keep_ticks
    
    return ticks


def plot_df_results(df1, df2, n_round, n_ticks, size, title1, title2, x_lab, y_lab, title=None, vmax1=None, vmax2=None):
    """
    Plot two df heatmaps as two subplots of one figure. They share x and y axis labels but have differing titles.

    :param df1: df to visualize
    :param df2: second df to visualize (often log of df1)
    :param n_round: number of axis tick decimal places to round to
    :param n_ticks: number of axis ticks to keep
    :param size: figure size
    :param title1: plot title for left (df1) panel
    :param title2: plot title for left (df2) panel
    :param x_lab: x axis label
    :param y_lab: y axis label
    :param title: filename to save figure as; its existence acts as a flag for saving/not saving this figure
    :param vmax1: cmap max cutoff value for df1
    :param vmax2: cmap max cutoff value for df2
    """
    # set y ticks: round and only display some to improve readability
    y_ticks = get_ticks(df1.index.values, n_round, n_ticks)
    fig, axs = plt.subplots(1, 2, figsize=(2*size, size))
    
    # plot first df as heatmap, using a cmap cutoff value if provided
    if vmax1 is None:
        sns.heatmap(df1, yticklabels=y_ticks, cbar_kws={'label': title1}, ax=axs[0], cmap='viridis')
    else:
        sns.heatmap(df1, yticklabels=y_ticks, cbar_kws={'label': title1}, ax=axs[0], cmap='viridis', vmax = vmax1)
    axs[0].set(xlabel=x_lab, ylabel=y_lab, title=title1)
    axs[0].invert_yaxis()
    
    # plot second df as heatmap, using a cmap cutoff value if provided
    if vmax2 is None:
        sns.heatmap(df2, yticklabels=y_ticks, cbar_kws={'label': title1}, ax=axs[1], cmap='viridis')
    else:
        sns.heatmap(df2, yticklabels=y_ticks, cbar_kws={'label': title1}, ax=axs[1], cmap='viridis', vmax = vmax2)
    axs[1].set(xlabel=x_lab, ylabel=y_lab, title=title2)
    axs[1].invert_yaxis()
    
    # if a filename is provided, save the figure with this filename; otherwise do not save
    if title is not None:
        plt.savefig(title)
