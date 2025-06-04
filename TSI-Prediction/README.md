# TSI-Prediction

This project focuses on AI-based gap filling of missing irradiance data for the two radiometers CLARA and DARA. The implemented models include unidirectional and bidirectioal LSTMs, neural networks (NN), temporal convolutional networks (TCN), and time series transfomers (PatchTST). 

## Project Overview

This project provides tools for:
-Preprocessing the radiometer data
-Imputing missing features and targets
-Training and evaluating machine learning models (LSTM, BILSTM, TCN, PatchTST, NN)
-Gap filling and forecasting of missing irradiance values

## Installation

To install the necessary packages use:

**For Conda environments:**
```sh
conda env create -f [environment.yml]

## Usage