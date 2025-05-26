import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import math
from utils import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time as t
from tcn import TemporalConvNet
from PatchTST import PTST
from types import SimpleNamespace

## HYPERPARAMETER ################################################################################

PATH_TRAIN = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_train_2021_to_2023.pkl'
PATH_TEST = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_test_2021_to_2023_postprocessed_mean.pkl'
TARGET_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Models/'
MODEL_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Models/PatchTST.pt'
# Setting SPLIT = 0 is equivalent to training on the full data available and filling in the found gaps
SPLIT = 0.2

# Network hyperparameters
input_size = 24
hidden_size = 128 # 128
output_size = 1
dropout = 0
num_layers = 5 
bidirectional = True
gap_filling = True
lstm = False
ff = False
tcn = False
window = 16 # 16

################################################################################################

## READ-IN #####################################################################################

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

## READ-IN #####################################################################################

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# generate train test split if specified or use new test data
df_train = pd.read_pickle(PATH_TRAIN)
X_train = df_train.drop(['IrrB'], axis = 1)  # Features for training
y_train = df_train['IrrB'] # Target

if SPLIT > 0:
    # Assuming a time-based split
    if gap_filling:
        X_train, X_test, y_train, y_test = create_gap_train_test_split(-1,-1,PATH_TRAIN)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)
    time_train = np.array(pd.DataFrame(X_train['TimeJD'])).flatten()
    time_test = np.array(pd.DataFrame(X_test['TimeJD'])).flatten()
    X_train = X_train.drop(['TimeJD'], axis = 1)
    X_test = X_test.drop(['TimeJD'], axis = 1)
else:
    df_test = read_pickle(PATH_TEST)
    X_test = df_test.drop(['IrrB', 'TimeJD'], axis = 1) # Features for gaps
    time_train = np.array(df_train['TimeJD'])
    time_test = np.array(df_test['TimeJD'])
    X_train = X_train.drop('TimeJD', axis = 1)

## DATA PREPARATION ############################################################################

# Normalize the data before proceeding
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale the features using MinMaxScaler
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(pd.DataFrame(y_train).values.reshape(-1,1))

# Convert into Torch Tensors to be able to work
X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)
X_test = torch.Tensor(X_test)

class train_dataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.y) - self.window_size + 1

    def __getitem__(self, idx):
        return self.X[idx:self.window_size + idx], self.y[self.window_size + idx-1]
    
class test_dataset(Dataset):
    def __init__(self, X, window_size):
        self.X = X
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size + 1

    def __getitem__(self, idx):
        return self.X[idx:self.window_size + idx]

train_dataset = train_dataset(X_train, y_train, window) # 8
test_dataset = test_dataset(X_test, window) # 8
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

################################################################################################

## MODELS ######################################################################################

# Create the LSTM network
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first = True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers

    def forward(self, input):
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(input, hidden)
        output = self.fc(lstm_out[:, -1, :])
        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Create the bidirectional LSTM network    
class BILSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first = True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.num_layers = num_layers

    def forward(self, input):
        batch_size = input.shape[0]
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(input, hidden)
        output = self.fc(lstm_out[:, -1, :])
        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size),
                torch.zeros(self.num_layers * 2, batch_size, self.hidden_size))
        
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers):
        super().__init__()
        layers = []
        in_features = input_size * window
        for i in range(num_layers - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.Dropout(dropout),
                nn.ReLU()
            ))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_inputs = input_size,
            num_channels = [hidden_size] * num_layers,
            kernel_size = 3,
            dropout = dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.tcn(x)
        return self.fc(x[:, : , -1])

class TST(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_layers):
        super().__init__()
        config = SimpleNamespace(
            enc_in = input_size,
            seq_len = window,
            pred_len = output_size,
            e_layers = num_layers,
            n_heads = 4,
            d_model = hidden_size,
            d_ff = hidden_size * 2,
            dropout = dropout,
            fc_dropout = dropout,
            head_dropout = dropout,
            individual = False,
            patch_len=4,
            stride=4,
            padding_patch=None,
            revin=False,
            affine=False,
            subtract_last=True,
            decomposition=False,
            kernel_size=3
        )
        self.model = PTST(config)
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.model(x).squeeze(1)
        return self.fc(x) 
###########################################y#####################################################

## GAP FILLIMNG ################################################################################
    
# Instantiate the desired model with the chosen parameters
if lstm:
    if not bidirectional:
        model = LSTM(input_size, hidden_size, output_size, dropout, num_layers)
    else:
        model = BILSTM(input_size, hidden_size, output_size, dropout, num_layers)
elif ff:
    model = NN(input_size, 2 * hidden_size, output_size, dropout, 2 * num_layers)
elif tcn:
    model = TCN(input_size, hidden_size, output_size, dropout, num_layers)
else:
    model = TST(input_size, hidden_size, output_size, dropout, num_layers)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

# Define the loss function and optimizer
criterion = nn.MSELoss()

# Generate predictions for the test dataset
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        # Forward pass
        outputs = model(inputs)
        pred = outputs.squeeze().tolist()
        if isinstance(pred, float):
            predictions.append(pred)
        else:
            predictions += pred

# Bring back to original scale
predictions = np.array(predictions).reshape(-1, 1)  # Reshape to 2D array
predictions = scaler_y.inverse_transform(predictions)

# Save outputs in desired folder
pd.DataFrame(predictions, columns=["Predicted"]).to_csv(TARGET_PATH + 'predicted_data.csv', index=False)

################################################################################################

## PLOT GENERATION #############################################################################

# Make sure the plotted data is in the original scale
irr_train = np.array(scaler_y.inverse_transform(y_train)).ravel()
irr_test = np.array(predictions).ravel()

# Create a single scatter plot with overlapping data points
plt.figure(figsize=(30, 6))

# Plot the original data in blue
sns.scatterplot(x = time_train, y = irr_train,  color = 'royalblue', label='Original train', s = 50)
if SPLIT > 0:
    sns.scatterplot(x = time_test, y = y_test, color='lightblue', label='Original test', s = 50)
sns.scatterplot(x = time_test[:(len(time_test)-window+1)], y = irr_test, color='deeppink', label='Predicted', s = 50)

# Add title and legend
plt.title('Overlay of Original and Predicted Data', fontsize = 32)
plt.legend(fontsize=20)
plt.xlabel('TimeJD', fontsize=12)
plt.ylabel('IrrB', fontsize=12)

# Save plot in the desired folder
plt.savefig(TARGET_PATH + 'output_plot.png')

time_train_dt = pd.to_datetime(time_train)
time_test_dt = pd.to_datetime(time_test)
half_years_train = time_train_dt.year.astype(str) + "-H" + ((time_train_dt.month > 6).astype(int) + 1).astype(str)
unique_half_years_train = sorted(half_years_train.unique())
half_years_test = time_test_dt.year.astype(str) + "-H" + ((time_test_dt.month > 6).astype(int) + 1).astype(str)
unique_half_years_test = sorted(half_years_test.unique())
half_years = sorted(set(unique_half_years_train) | set(unique_half_years_test))

fig, axes = plt.subplots(len(half_years), 1, figsize=(30, 6 * len(half_years)), sharey=True)
start_train, end_train = 0, 0
start_test, end_test = 0, 0 
for i, hy in enumerate(half_years):
    ax = axes[i]
    train_mask = half_years_train == hy
    sns.scatterplot(x = time_train[train_mask], y = irr_train[train_mask],  color = 'royalblue', label='Original train', s = 50, ax = ax)
    test_mask = half_years_test == hy
    if SPLIT > 0:
        sns.scatterplot(x = time_test[test_mask], y = y_test[test_mask], color='lightblue', label='Original test', s = 50, ax = ax)
    tmp = time_test[:len(irr_test)]
    sns.scatterplot(x = tmp[test_mask[:len(irr_test)]], y = irr_test[test_mask[:len(irr_test)]], color='deeppink', label='Predicted', s = 50, ax = ax)
    ax.set_title(f'Predictions for {hy}', fontsize=20)
    ax.set_xlabel('TimeJD')
    ax.set_ylabel('IrrB')
    ax.legend()

plt.tight_layout()
plt.savefig(TARGET_PATH + 'output_plot_by_half_year.png')

################################################################################################

## ERROR MEASURE ###############################################################################

# This is only available in case the code is being run with a split

if SPLIT > 0:
    mse = mean_squared_error(y_test[:(len(time_test)-window+1)], irr_test)
    print(f"Mean Squared Error on the test split: {mse}")

################################################################################################