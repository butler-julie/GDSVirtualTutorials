# Evidential Deep Learning

## Overview
Presented by: Ayush Khot (University of Illinois at Urbana-Champaign)

Date: October 3, 2025

## Summary
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/butler-julie/GDSVirtualTutorials/blob/main/100325_UncertaintyQuantification/EDL.ipynb)

Machine learning (ML) has become vital in modern high-energy physics, enabling the analysis of vast amounts of data obtained from complex detector systems. A key open problem is uncertainty quantification (UQ), since current Bayesian methods are accurate but often computationally expensive and time-consuming. This tutorial introduces evidential deep learning (EDL), an emerging method that treats learning as an evidence acquisition process designed to provide confidence about test data. Using publicly available datasets for jet classification benchmarking from the Large Hadron Collider, we will demonstrate how to train and optimize EDL models, compare their uncertainty estimates to Bayesian ensembles, and visualize how uncertainties distribute across jet classes and latent spaces. We will also explore EDL’s potential for anomaly detection in collider data.

The tutorial is accessible using the Colab link at the top, or it can be run locally in an environment with the libraries listed in the `requirements.txt`.
