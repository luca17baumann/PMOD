import astropy
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
import pandas as pd
import os
from utils import *

## HYPERPARAMETERS ################################################################################

SOURCE_L1_PATH = 'dummy'
SOURCE_L2_PATH = 'dummy'

PATH_FEATURES = '/Users/juliabarth/Desktop/DSLab/combined_CLARA_all.pkl'

TARGET_PATH = "combined_data.pkl"

time = "Irradiance TimeJD"
target = "irradiance_B [W.m-2]"
target2 = "irradiance_A [W.m-2]"
target3 = "irradiance_C [W.m-2]"

###################################################################################################

def list_files_in_subfolders(folder_path):
    ''' Expected folder structure:
    
        folder_path/
        ├── month1/                  
        │   └── day1-file
        │   └── day2-file    
        │   └── ...
        ├── month2/ 
        ... 
    '''
    file_paths = []
    for month in os.listdir(folder_path):
        try:
            if(month != ".DS_Store"):
                print(month)
                for day in os.listdir(folder_path + "/" + month):
                    if(day != ".DS_Store"):
                        print(day)
                        file_paths.append(folder_path + "/" + month + "/" + day)
        except:
            print("No files in this folder")
    print(len(file_paths))
    return file_paths


def read_file_level1(path: str, features_level_1: list) -> pd.DataFrame:
    """Read as file from level1 and returns a dataframe indexed by time
    
    Args:
        path (str) -> where to read the files from
        features_level_1 (list) -> a list of the desired features

    Returns:
        pd.DataFrame
    """
    all_data = fits.open(path)

    # only keep time until minute
    TimeJD = all_data[8].data.field("Timestamp")
    TimeJD = Time(TimeJD, format="isot")
    TimeJD = TimeJD.strftime("%Y-%m-%d %H:%M")

    data = {"TimeJD": TimeJD}
    
    try: 
        for feature in features_level_1:
            data[feature] = all_data[8].data.field(feature)
        dtypes = {feature: float for feature in features_level_1}
        data = pd.DataFrame(data).astype(dtypes)

        # groupby TimeJD and take the mean (there are multiple measurements per minute)
        data["TimeJD"] = pd.to_datetime(data["TimeJD"])
        data_per_minute = data.groupby("TimeJD").mean()
        data_per_minute.reset_index(inplace=True)

        return data_per_minute
    
    except:
        print(f'File {path} did not contained (some of) the columns specified and was excluded.')
        return 


def concatenate_level1(paths: list, features_level_1: list) -> pd.DataFrame:
    """Concatenates all files from level1 and returns a dataframe indexed by time

    Args:
        paths (list)
        features_level_1 (list) -> a list of the desired features

    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame()
    for path in paths:
        next_day = read_file_level1(path, features_level_1)
        df = pd.concat([df, next_day])
    return df


def read_file_level2(path: str, time:str, target:str, target2: str = None, target3: str = None) -> pd.DataFrame:
    """Reads file from level2 and returns a dataframe indexed by time

    Args:
        path (str): Path to the file
        time (str): Name of the time field
        target (str): Name of the target field
        target2 (str, optional): Name of the second target field (default: None)
        target3 (str, optional): Name of the third target field (default: None)
        
    Returns:
        pd.DataFrame
    """
    all_data = fits.open(path)
    TimeJD = all_data[1].data.field(time)  # time in Julian day
    TimeJD = Time(TimeJD, format="jd", scale="utc").iso
    
    IrrB = all_data[1].data.field(target)  # estimated TSI for cavity B
    
    if target2 is not None:
        IrrA = all_data[1].data.field(target2)  # estimated TSI for cavity A
    else:
        IrrA = None
    
    if target3 is not None:
        IrrC = all_data[1].data.field(target3)  # estimated TSI for cavity C
    else:
        IrrC = None
    

    data = {"TimeJD": TimeJD, "IrrA": IrrA, "IrrB": IrrB, "IrrC": IrrC}

    dtypes = {
        "IrrA": float,
        "IrrB": float,
        "IrrC": float,
    }
    
    data = pd.DataFrame(data).astype(dtypes)
    
    if target2 is None:
        data.drop(columns=["IrrA"], inplace=True)
    if target3 is None:
        data.drop(columns=["IrrC"], inplace=True)

    # groupby TimeJD and take the mean (there are multiple measurements per minute). Takes also care of the NaN values
    data["TimeJD"] = pd.to_datetime(data["TimeJD"], errors="coerce")
    data.dropna(subset=["TimeJD"], inplace=True)
    data["TimeJD"] = data["TimeJD"].dt.floor("T")
    data_per_minute = data.groupby("TimeJD").mean()
    data_per_minute.reset_index(inplace=True)

    return data_per_minute


def concatenate_level2(paths: list, time:str, target:str, target2: str = None, target3: str = None) -> pd.DataFrame:
    """Concatenates all files from level2 and returns a dataframe indexed by time

    Args:
        paths (list)
        time (str): Name of the time field
        target (str): Name of the target field
        target2 (str, optional): Name of the second target field (default: None)
        target3 (str, optional): Name of the third target field (default: None)
        
    Returns:
        pd.DataFrame
    """
    df = pd.DataFrame()
    for path in paths:
        next_day = read_file_level2(path, time, target, target2, target3)
        df = pd.concat([df, next_day])
    return df


def merge_level1_level2(df_level1: pd.DataFrame, df_level2: pd.DataFrame) -> pd.DataFrame:
    """Merges dataframes from level1 and level2 and returns a dataframe indexed by time

    Args:
        df_level1 (pd.DataFrame)
        df_level2 (pd.DataFrame)

    Returns:
        pd.DataFrame
    """
    df = df_level1.merge(df_level2, on="TimeJD", how = "left")
    return df


def main():
    
    # LEFT JOIN (including gap data)
    
    features_level_1 = read_file(PATH_FEATURES)
    paths_files_level1 = list_files_in_subfolders(SOURCE_L1_PATH)
    paths_files_level2 = list_files_in_subfolders(SOURCE_L2_PATH)

    # Read all files
    df_level1 = concatenate_level1(paths_files_level1, features_level_1)
    df_level2 = concatenate_level2(paths_files_level2, time, target, target2, target3)

    # Combine the data from level1 and level2 and save to pickle file
    df_combined = merge_level1_level2(df_level1, df_level2)
    df_combined.to_pickle(TARGET_PATH)
    
    print("Merging: complete. Data file in ", TARGET_PATH)

if __name__ == "__main__":
    main()