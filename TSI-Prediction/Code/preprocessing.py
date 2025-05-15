import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import math
from utils import *
import time as t

## HYPERPARAMETER ################################################################################

t0 = t.time()

# Read-in:
PATH_DATA = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/combined_data.pkl'
PATH_FEATURES = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/combined_data.pkl'
TARGET_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/'
IMAGE_PATH = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Images/'

# Preprocessing:
OUTLIER_UPPER = 1370
OUTLIER_LOWER = 1356
THRESHOLD = 3

################################################################################################

## READ-IN #####################################################################################

df = pd.read_pickle(PATH_DATA)
feature_list = read_pickle(PATH_FEATURES)

missing_features = [feature for feature in feature_list if feature not in df.columns]
if missing_features:
    print("Warning: The following features are not present in the DataFrame:", missing_features)

columns = [i for i in df.columns if i in feature_list]
df = df[columns]

print("READ-IN: complete.")

## PREPROCESSING #################################################################################

df = df.sort_values(by=['TimeJD'])

# Safety tests and data validation checks

missing_values = df.isnull().sum()
if missing_values.any():
    print("Warning: Missing values found. Handling missing values if not in IrrB...")
    columns = df.columns[df.columns != 'IrrB']
    # Deleting rows with missing values (do not consider IrrB in light of gap filling)
    df = df.dropna(subset=columns)
    # Creating a fully clean dataset as well
    df_clean = df.dropna()
else:
    print("SANITY CHECK: No missing values found.")

# Range Validation
# Assuming IrrB should be within [OUTLIER_LOWER, OUTLIER_UPPER] range
out_of_range_indices = df[(df['IrrB'] < OUTLIER_LOWER) | (df['IrrB'] > OUTLIER_UPPER)].index
if not out_of_range_indices.empty:
    print("Warning: Values outside expected range found. Handling out-of-range values...")
    # Handling out-of-range values
    df = df[((df['IrrB']<OUTLIER_UPPER) & (df['IrrB']>OUTLIER_LOWER)) | df['IrrB'].isna()]

# Unique Values
duplicates = df.duplicated()
if duplicates.any():
    print("Warning: Duplicates found. Removing duplicates.")
    df = df.drop_duplicates()
    
################################################################################################

## CORRELATION PLOT ############################################################################

# Generate data and set axis
data = df_clean.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(10, 8))

# Define colormap and generate correlation plot
cmap = plt.cm.coolwarm
image = ax.matshow(data, cmap=cmap)

# Add colorbar, set ticks, labels and the title
cb = plt.colorbar(image)
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))
ax.set_xticklabels(data.columns, rotation=45, fontsize=8, ha='left')
ax.set_yticklabels(data.columns, fontsize=8)
plt.title('Correlation Matrix for CLARA data', fontsize=20)

# Add text annotations to see individual correlation values
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax.text(j, i, "{:.2f}".format(data.iloc[i, j]), ha='center', va='center', color='black', fontsize=6)

# Save plot in the desired folder
plt.savefig(IMAGE_PATH + 'correlation_plot.png')

################################################################################################

## OUTLIER REMOVAL #############################################################################

# Checking for Outliers and removing them with rolling median
df.reset_index(inplace=True)

# Set the size of the rolling window for calculating the median and compute median and deviation
window_size = 5
rolling_median = df['IrrB'].dropna().rolling(window=window_size).median()
deviation = abs(df['IrrB'].dropna() - rolling_median)

# Eliminate outliers beyond threshold
outliers_mask = deviation >= THRESHOLD
original_length = len(df["IrrB"])
outlier_mask_df = outliers_mask.to_frame().reset_index()
outlier_indx = outlier_mask_df[outlier_mask_df['IrrB']]['index']
noutlier_indx = outlier_mask_df[~outlier_mask_df['IrrB']]['index']

# Create a new DataFrame containing only the outliers
outliers = df.drop(noutlier_indx, axis=0)
if len(outliers) > 0:
    print("Warning: Outliers found. Handling outliers...")

# Remove the outliers from the original DataFrame
df = df.drop(outlier_indx, axis=0)
print(f'Removing: {original_length-len(df["IrrB"])} datapoints')

# Create plots for visualizing the original data and outliers
fig, axes = plt.subplots(2,1, figsize=(15, 8))
sns.scatterplot(x=df["TimeJD"], y=df["IrrB"], ax = axes[0]).set(title=f'Scatterplot of IrrB Median - threshold: {THRESHOLD}')
sns.scatterplot(x=outliers['TimeJD'], y=outliers['IrrB'], ax = axes[1], color='red' )

# Save plot in the desired folder
plt.savefig(IMAGE_PATH + 'outlier_plot.png')

################################################################################################

## GAP FINDING #################################################################################

# Separate the data in train and test based on missing values
df_train = df[df['IrrB'].notna()].copy()
df_test = df[df['IrrB'].isna()].copy()

# Set minimum treshold to fill and find gaps bigger than that
gap_threshold = pd.Timedelta(days=1)
df_train['gap'] = df_train['TimeJD'].diff()
large_gaps_mask = (df_train['gap'] > gap_threshold)
selected_gaps = df_train[large_gaps_mask]

# Resample all data and test data
time_interval = pd.to_timedelta('15 minutes')
df = df.drop('index', axis = 1)
df.set_index('TimeJD', inplace=True)
df_test = df_test.drop('index',axis = 1)
df_test.set_index('TimeJD', inplace=True)
df_resampled = df.resample(time_interval).mean().copy()

# Drop NAs due to the averaging
df_resampled.dropna(subset=df_test.columns.difference(['IrrB']), inplace=True)
df_test = df_resampled[df_resampled['IrrB'].isna()].copy()

# Select the desired rows
sampled_rows = pd.DataFrame()
for index, gap in selected_gaps.iterrows():
    end_time = gap['TimeJD']
    start_time = end_time - gap['gap']

    current_rows = df_test.loc[start_time:end_time].copy()
    sampled_rows = pd.concat([sampled_rows, current_rows])

# Reset the index for the final result
sampled_rows.reset_index(inplace=True)
df_train.drop('gap', axis = 1, inplace = True)
df_train.drop('index', axis = 1, inplace = True)

df_resampled = df_resampled.reset_index()
df_resampled['TimeJD'] = pd.to_datetime(df_resampled['TimeJD'])
df_test = df_test.reset_index()
df_test['TimeJD']= pd.to_datetime(df_test['TimeJD'])

################################################################################################

## SAVE PREPROCESSED DATA ######################################################################

df_train.to_pickle(TARGET_PATH + 'df_train.pkl')
df_test.to_pickle(TARGET_PATH + 'df_test.pkl')

t1 = t.time()
time_elapsed = t1 - t0

print("Execution time: ", int(time_elapsed / 60), " Minutes ", int(time_elapsed % 60)," Seconds.")

################################################################################################