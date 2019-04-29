============================
diffusive_distinguishability
============================


.. image:: https://readthedocs.org/projects/diffusive-distinguishability/badge/?version=latest
        :target: https://diffusive-distinguishability.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Simulation of homogeneous diffusion, bayesian estimation of underlying diffusion constant and analysis of distinguishability between diffusivities (ndim_homogeneous_distinguishability.py in diffusive_distinguisbaility)

Jupyter notebook providing examples of how to interact with submodules of above .py, and example analysis (ndim_diffusion_analysis_tutorial.ipynb)


* Free software: Allen Institute Software License

* Documentation: https://diffusive-distinguishability.readthedocs.io.


Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

Associated text
---------------
Below is a working abstract for the 'Bayesian detection of diffusive heterogeneity' project found in this repo: 

**Abstract**
Cells are crowded and spatially heterogeneous, complicating the transport of organelles,proteins and other substrates. The diffusion constant partially characterizes dynamicsin complex cellular environments but, when taken on a per-cell basis, fails to capturespatial dependence of diffusivity. Measuring spatial dependence of diffusivity ischallenging because temporally and spatially finite observations offer limitedinformation about a spatially varying stochastic process. We present a Bayesianframework that estimates diffusion constants from single particle trajectories, andpredicts our ability to distinguish differences in diffusion constants, conditional on howmuch they differ and the amount of data collected.

**Introduction**
Diffusion is essential for the intracellular transport of organelles, proteins and substrates, and is commonly characterized through analyses of single particle tracking (SPT) in live-cell images. While powerful analyses from SPT have indicated the complexity of transport in live cells, the spatial variation of the diffusion constant remains poorly characterized. This can be attributed to challenges in disentangling effects of biological heterogeneity and limited sampling of a stochastic process. To address these challenges, we developed a Bayesian framework to estimate a posterior distribution of the possible diffusion constants underlying SPT dynamics. This framework can be used to generate a look-up table predicting the detectability of differences in diffusion constants, conditional on the ratio of their values and amount of trajectory data collected.

**Materials and methods**
We simulate particle diffusion in a range of homogeneous diffusion constants, and digest the resulting trajectories into frame-to-frame displacements. Using an inverse-gamma conjugate prior, we make the conservative guess that any order of magnitude of diffusion constant is equally likely. The set of displacements in a single trajectory are used to generate a posterior inverse-gamma distribution estimating the probability that any given diffusion constant was used to generate the trajectory. This distribution peaks near the true diffusion constant and has a width corresponding to the confidence interval of our estimate, which is largely determined by the trajectory length. Given a pair of posteriors derived for trajectories with differing underlying diffusion constants, we can characterize their similarity by computing the Kullback-Leibler divergence. This metric acts as a single-value estimation of how well we can analytically distinguish that trajectories were generated from different diffusion constants. For longer trajectory lengths, stochastic variations will be less dominant, increasing distinguishability.

**Results**
To assess the conditional feasibility of computationally detecting differences in diffusivity, we generate a landscape of the KL divergence between posteriors generated from pairs of simulations, with varying trajectory lengths and differences in diffusivity. To further correct for stochastic variations in simulations, the KL divergence reported for each entry in the landscape is the mean value from thousands of replicates. We find that, using this method, diffusivities differing by a factor of 1.5 or more can be easily distinguished when at least 50 timepoints are reported for each trajectory. This landscape offers a look-up table for estimating the number of frames that must be acquired experimentally to distinguish diffusivities to a desired precision. This framework could therefore play a valuable role in describing the feasibility of and requirements for experiments addressing the spatial heterogeneity of the intracellular diffusive environment. To address the affects of static localization error of punctate objects from microscopy images, we included Gaussian error to the particle location at each point in its trajectory. The standard deviation of this Gaussian determines the amount of localization error applied. Now, error in the ability to detect the underlying diffusion constant is a compound error due to the affects of both localization error and error in Bayesian estimation of the posterior maximum.

**Conclusion**
The spatial heterogeneity of diffusion may have major impacts in the transport ofessential cellular substrates but remains largely uncharacterized. To shed light on thefeasibility of resolving spatial from stochastic drivers of diffusive heterogeneity intrajectory data, we developed a framework for predicting our ability to detect differences in diffusivity, conditional on the amount of experimental data collected. Our framework can therefore be used to inform the design of experiments aimed to characterize the spatial dependence of diffusivity across cells.


Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
