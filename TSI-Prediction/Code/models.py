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

# TODO: CHECK PREPROCESSING AGAIN BECAUSE OF INDEX WHICH IS CONTAINED IN TRAIN BUT NOT IN TEST

## HYPERPARAMETER ################################################################################

PATH_TRAIN = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_train.pkl'
PATH_TEST = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data//df_test.pkl'
TARGET_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction//Models/'
# Setting SPLIT = 0 is equivalent to training on the full data available and filling in the found gaps
SPLIT = 0

# Network hyperparameters
input_size = 30
hidden_size = 128
output_size = 1
learning_rate = 3e-3
num_epochs = 100
dropout = 0.0
num_layers = 2
bidirectional = False

################################################################################################

## READ-IN #####################################################################################

# generate train test split if specified or use new test data
df_train = pd.read_pickle(PATH_TRAIN)
X_train = df_train.drop(['IrrB', 'TimeJD'], axis = 1)  # Features for training
y_train = df_train['IrrB'] # Target

if SPLIT > 0:
    # Assuming a time-based split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=False)

else:
    df_test = read_pickle(PATH_TEST)
    X_test = df_test.drop(['IrrB', 'TimeJD'], axis = 1) # Features for gaps

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

torch.manual_seed(42)

train_dataset = train_dataset(X_train, y_train, 16) # 8
test_dataset = test_dataset(X_test, 16) # 8
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
        return (torch.randn(self.num_layers, batch_size, self.hidden_size),
                torch.randn(self.num_layers, batch_size, self.hidden_size))

# Create the bidirectional LSTM network    
class BILSTM(nn.Module):
    pass

################################################################################################

## GAP FILLIMNG ################################################################################
    
# Instantiate the desired model with the cosen parameters
if not bidirectional:
    model = LSTM(input_size, hidden_size, output_size, dropout, num_layers)
else:
    model = BILSTM(input_size, hidden_size, output_size, dropout, num_layers)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.8f}'
          .format(epoch+1, num_epochs, train_loss / len(train_dataloader)))

# Set the model to evaluation mode
model.eval()
0
# Generate predictions for the test dataset
predictions = []
with torch.no_grad():
    for inputs in test_dataloader:
        # Forward pass
        outputs = model(inputs)
        predictions += outputs.squeeze().tolist()

# Bring back to original scale
predictions = scaler_y.inverse_transform(predictions)

# Save outputs in desired folder
predictions.to_csv(TARGET_PATH + 'predicted_data.csv', index = False)

################################################################################################

## PLOT GENERATION #############################################################################

# Convert data in the appropriate format
time_train = np.array(df_train['TimeJD'])
time_test = np.array(df_test['TimeJD'])

# Make sure the plotted data is in the original scale
irr_train = np.array(scaler_y.inverse_transform(y_train)).ravel()
irr_test = np.array(predictions)

# Create a single scatter plot with overlapping data points
plt.figure(figsize=(30, 6))

# Plot the original data in blue
sns.scatterplot(x = time_train, y = irr_train,  color = 'royalblue', label='Original train', s = 50)
if SPLIT > 0:
    sns.scatterplot(x = time_test, y = y_test, color='lightblue', label='Original test', s = 50)
sns.scatterplot(x = time_test, y = irr_test, color='deeppink', label='Predicted', s = 50)

# Add title and legend
plt.title('Overlay of Original and Predicted Data', fontsize = 32)
plt.legend(fontsize=20)
plt.xlabel('TimeJD', fontsize=12)
plt.ylabel('IrrB', fontsize=12)

# Save plot in the desired folder
plt.savefig(TARGET_PATH + 'output_plot.png')

################################################################################################

## ERROR MEASURE ###############################################################################

# This is only available in case the code is being run with a split
mse = mean_squared_error(y_test, irr_test)
print(f"Mean Squared Error on the test split: {mse}")

################################################################################################