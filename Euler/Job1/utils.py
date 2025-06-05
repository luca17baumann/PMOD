import pandas as pd
import pickle
import numpy as np
np.random.seed(1)


def read_file(file_path):
    ''' Expected structure of the file: 
        one column name per line (no header)
    '''
    try:
        with open(file_path, 'r') as file:
            column_names = [line.strip() for line in file]
            return column_names
        
    except FileNotFoundError:
        print("File not found.")
        return []
    
def read_pickle(file_path):
    '''Read file in binary format
    '''
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print("File not found.")
        return []

def create_gap_train_test_split(gap, period, data_path, date = None):
    '''Function to create a train- and testdataset for gapfilling. For entire dataset with gaps ranging from 1 to 6 days use
    gap = -1, period = -1, and the respective datapath
    Inputs:
        gap: indicates how many days the gap should include (for example gap = 1 means the gap is a single day) use -1 for 1 to 6 day gaps
        period: indicates the reconstruction period in months (for example period = 1 means there is halve a month before the gap and halve a month afterwards) use -1 for entire data
        data_path: path from where the data can be loaded.
        date: date of the center which should be predicted
    Output:
        X_train, y_train: Training data
        X_test, y_test: Test data
    '''
    data = pd.read_pickle(data_path)
    
    if period > 0 and gap > 0:
        if not date: 
            median_date = data['TimeJD'].median().date() 
        else:
            median_date = pd.to_datetime(date).date()
        days_window = int((period * 30) / 2)
        gap_window = int(gap / 2)
    
        if gap_window >= days_window:
            raise ValueError("Not enough data to reconstruct gap increase period")
        # get data for training
        mask = get_mask(median_date, days_window, data, False)
        masked_data = data[mask]
        
        # get test data
        test_mask = get_mask(median_date, gap_window, masked_data, gap % 2 == 0)
        test = masked_data[test_mask]
        train = masked_data[~test_mask]
    elif period == -1 and gap == -1:
        dates = sorted(data['TimeJD'].dt.date.unique())
        split_indices = [int(i * len(dates) / 6) for i in range(7)]
        date_splits = [dates[split_indices[i]:split_indices[i+1]] for i in range(6)]
        arr = np.random.permutation(np.arange(1, 7))
        ctr = 0
        train = []
        test = []
        for split in date_splits:
            mask = data['TimeJD'].dt.date.isin(split)
            masked_data = data[mask]
            median_date = masked_data['TimeJD'].median().date()
            curr_gap = arr[ctr]
            ctr += 1
            gap_window = int(curr_gap / 2)
            # get test data
            test_mask = get_mask(median_date, gap_window, masked_data, curr_gap % 2 == 0)
            test.append(masked_data[test_mask])
            train.append(masked_data[~test_mask])
        train = pd.concat(train, axis=0)
        test = pd.concat(test, axis=0)
    else:
        raise ValueError("Choose valid gap length and reconstruction period.")
    
    X_train = train.drop(['IrrB'], axis = 1)
    y_train = train['IrrB']
    X_test = test.drop(['IrrB'], axis = 1)
    y_test = test['IrrB']
    return X_train, X_test, y_train, y_test

def get_mask(median_date, window, data, even):
    '''Function to get the mask of the windowed data.
    Inputs:
        median_date: data around which the window is centered
        window: days from median_date to start/end of window
        data: data for which the mask should be returned
        even: Boolean to indicate if there is an even number of points in the window
    Outputs:
        mask: mask which can be used to get windowed data
    '''
    dates = sorted(data['TimeJD'].dt.date.unique())
    if median_date in dates:
        idx = dates.index(median_date)
    else:
        raise ValueError("median_date is not present in data")
    
    window_dates = []
    if even:
        for i in range((idx-window+1),idx+window+1):
            if 0 <= i < len(dates):
                window_dates.append(dates[i])
            else:
                raise ValueError("Window ranges out of data decrease window")
    else:
        for i in range((idx-window),idx+window+1):
            if 0 <= i < len(dates):
                window_dates.append(dates[i])
            else:
                raise ValueError("Window ranges out of data decrease window")
    mask = data['TimeJD'].dt.date.isin(window_dates)
    return mask

# PATH_TRAIN = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/df_train.pkl'
# X_train, X_test, y_train, y_test = create_gap_train_test_split(1,1,PATH_TRAIN)