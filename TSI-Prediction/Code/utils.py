import pandas as pd
import pickle


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

def create_gap_train_test_split(gap, period, data_path):
    '''Function to create a train- and testdataset for gapfilling.
    Inputs:
        gap: indicates how many days the gap should include (for example gap = 1 means the gap is a single day)
        period: indicates the reconstruction period in months (for example period = 1 means there is halve a month before the gap and halve a month afterwards)
        data_path: path from where the data can be loaded.
    Output:
        X_train, y_train: Training data
        X_test, y_test: Test data
    '''
    data = pd.read_pickle(data_path)
    median_date = data['TimeJD'].median().date() 
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