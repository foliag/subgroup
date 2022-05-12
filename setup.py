#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import setuptools
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

with open("README.txt","r") as fh:
    long_description = fh.read()



setuptools.setup(
    name="heteroverlap",
    version="0.0.1",
    author="ZiyeLuo",
    author_email="2017100369@ruc.edu.cn",
    description="Regression-based heterogeneity analysis to identify overlapping subgroup structure in high-dimensional data",
    long_description_content_type = "text/plain",
    long_description=long_description,
    url="https://github.com/foliag/subgroup",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.21.2",
                      "openpyxl>=3.0.9",
                      "pandas>=1.3.3",
                      "scikit-learn>=1.0.1",
                      "scipy>=1.7.1",
                      "seaborn>=0.11.2"
                      
        ],
    classifiers=[
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",]
    
    )