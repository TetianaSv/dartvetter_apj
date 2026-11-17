# COMP1818 coursework extension (Tetiana Svynar, studentID 001390358)

This repository is a fork of DART-Vetter (Fiscale et al. 2025, AJ 170, 73), extended for the COMP1818 AI Applications coursework "From Kepler to TESS: Calibrating DART-Vetter Under Domain Shift".

The authors' model and training code (m1_class.py, train_test_cnn_class.py) are unchanged; only the paths in config_cnn.yaml were edited. Everything below the line "Original README" is the authors' text.

## What was added

build_dataset.py - Downloads Kepler DR24 / TESS TOI light curves (NASA Exoplanet Archive + MAST, Lightkurve) and builds 201-bin global views

split_dataset.py - Merges shards and makes an 80/20 train/test split by star

clip_views.py - Clips global-view values to [-3, 3] (removes extreme outliers)

diagnose_model.py - Diagnoses the trained model and recalculates BatchNorm statistics; saves the final model \*\_bnfix.pt

baselines.py - Random coin, majority class, Logistic Regression, Random Forest and DART-Vetter on the Kepler test set

evaluate_tess.py - Applies the Kepler-trained models to TESS without retraining

dfo.py - Dispersive Flies Optimisation (al-Rifaie 2014)

calibrate_conformal.py - Platt scaling fitted with DFO (checked against L-BFGS-B), split conformal prediction, Monte Carlo study of the calibration-set size

bn_experiments.py, mc_dropout.py - Exploratory diagnostics (precise BatchNorm, MC dropout); not needed to reproduce the results

## Data and trained model

Data, trained models, predictions, results and plots (not stored on GitHub because of their size): OneDrive: [COMP1818_DART-Vetter](https://uogcloud-my.sharepoint.com/:f:/g/personal/ts4179i_gre_ac_uk/IgDU6y1Q_3oWSKbKI3QAlQuAAdXYa7qFbXXBur9btSbO_T4?e=SHe8Ei) (file dartvetter_coursework_data.zip). After downloading, unzip the archive in the project root.

## Environment

Tested with Python 3.11 on macOS.

conda create -n dartvetter python=3.11
conda activate dartvetter
pip install -r requirements_coursework.txt

All commands are run from the project root. Random seed: 42 everywhere.
python baselines.py # Table 2 (Kepler part)
python evaluate_tess.py # Table 2 (TESS part), TESS breakdown by disposition
python calibrate_conformal.py # Table 3, Table 4, Figure 1, Figure 2

# Original README

# DART-Vetter: A Convolutional Neural Network-based model for Exoplanets Detection

## Overview

This repository contains the code for implementing DART-Vetter, a Convolutional Neural Network (CNN) designed to classify Threshold Crossing Events (TCEs) as either planetary transits or false positives. The model is tailored for photometric data collected from space-based missions like NASA's Kepler and Transiting Exoplanet Survey Satellite (TESS).

## Features

- **Binary/Multi-class classification**: in binary classification mode, the model distinguishes planetary transits and contact eclipsing binaries from other phenomena like stellar variability, instrumental artifacts and non-contact eclipsing binaries. With a simple change in the configuration file and using a suitable dataset, the model is able to solve multi-class classification tasks.
- **Configurable architecture**: the network architecture is highly customizable, with parameters defined in an external configuration file.
- **Ease of use**: includes scripts for training, testing, and evaluating the model on custom datasets.
- **Visualization tools**: generates performance metrics like PR curves and ROC curves.

## Usage

- **Configuration**: the model parameters and dataset paths are defined in the `config_cnn.yaml` file. Update the file with your dataset paths and desired hyperparameters;
- **Input data**: you can download the input global views from the link provided in `link_to_input_data.txt`. Given a csv file, each row has this structure: (**x**,y). **x** is the 201-dimensional input vector and y is its relative label;
- **Training**: use `train_test_cnn_class.py` to train and test the model;
- **Model architecture**: the CNN architecture is defined in m1_class.py;
- **Output**: Model checkpoints, logs and plots will be saved to the directory specified in the configuration file.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/username/dart-vetter.git
cd dart-vetter
```

2. Install required Python packages:

```bash
pip install -r requirements.txt
```

If case of any error during step 2, try to run this command:

```bash
conda env create -f dartvetter_apj.yml
```
