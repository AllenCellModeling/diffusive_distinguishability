============================
diffusive_distinguishability
============================


.. image:: https://readthedocs.org/projects/diffusive-distinguishability/badge/?version=latest
        :target: https://diffusive-distinguishability.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
.. image:: https://zenodo.org/badge/183488372.svg
   :target: https://zenodo.org/badge/latestdoi/183488372


Simulation of homogeneous isotropic diffusion, bayesian estimation of underlying diffusion constant and analysis of distinguishability between diffusivities 


Getting Started
---------------

The python package ``ndim_homogeneous_distinguishability.py`` contains the meat of this project, as a set of functions which can be used to:

1. Simulate diffusive trajectories (diffusion with a homogeneous isotropic diffusion constant)
2. Use Bayesian inference to estimate the diffusion constant used to generate a trajectory by producing a posterior diffusivity distribution
3. Analyze the dependence of diffusivity estimation error, and the ability to distinguish between trajectories with differing diffusivities, conditional on model parameters

This repo also includes:

1. ``ndim_diffusion_analysis_tutorial.ipynb``: Jupyter notebook providing examples of how to use these functions, as well as some of our own analysis of diffusivity distinguishability. This includes the function calls used to generate some ``"Results"`` figures in our manuscript
2. ``figure_production.ipynb``: Jupyter notebook used to generate the remainder of our computationally derived manuscript figures, provided for reproducibility
3. ``test_overestimation.ipynb``: Jupyter notebook containing a toy model quantifying the relative impact of localization error on diffusion estimates conditional on number of spatial dimensions
4. ``saved_data/``: Directory containing some example pre-calculated datasets (in the form of pickled dataframes), generated in our Jupyter notebook example analyses described above. Text files are included with the datasets, specifying the parameters used in their generation
5. ``figures/``: Directory storing all of our computationally derived manuscript figured, stored as image files


* Free software: Allen Institute Software License

* Documentation: https://diffusive-distinguishability.readthedocs.io.


Installation
------------

PyPI installation not available at this time, please install using git.:

``pip install git+https://github.com/AllenCellModeling/diffusive_distinguishability.git``

Support
-------
We are not currently supporting this code, but simply releasing it to the community AS IS but are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.


Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
