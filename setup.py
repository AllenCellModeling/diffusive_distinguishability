#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'futures']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', 'pytest-cov', 'pytest-raises', ]

dev_requirements = [
    'bumpversion>=0.5.3',
    'wheel>=0.33.1',
    'flake8>=3.7.7',
    'tox>=3.5.2',
    'coverage>=5.0a4',
    'Sphinx>=2.0.0b1',
    'twine>=1.13.0',
    'pytest>=4.3.0',
    'pytest-cov==2.6.1',
    'pytest-raises>=0.10',
    'pytest-runner>=4.4'
]

extra_requirements = {
    'test': test_requirements,
    'setup': setup_requirements,
    'dev': dev_requirements
}

setup(
    author="Julie Cass",
    author_email='juliec@alleninstitute.org',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Allen Institute Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Simulation of homogeneous diffusion, bayesian estimation of underlying diffusion constant and analysis of distinguishability between diffusivities",
    entry_points={
        'console_scripts': [],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='diffusive_distinguishability',
    name='diffusive_distinguishability',
    packages=find_packages(include=['diffusive_distinguishability']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    extras_require=extra_requirements,
    url='https://github.com/jcass11/diffusive_distinguishability',
    version='0.1.0',
    zip_safe=False,
)
