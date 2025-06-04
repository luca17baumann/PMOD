import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
from shapely.geometry import Polygon, MultiPolygon
from utils import *
import time as t

## HYPERPARAMETER ################################################################################

PATH_DATA = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA/'
PATH_COMBINED = '/Users/luca/Desktop/Internship/PMOD/TSI-Prediction/Data/CLARA_combined.pkl'

if not os.path.exists(PATH_COMBINED):
    df = pd.DataFrame()
    files = sorted([f for f in os.listdir(PATH_DATA) if os.path.isfile(os.path.join(PATH_DATA, f))])
    # combine the data
    for file in files:
        tmp = pd.read_pickle(PATH_DATA + file)
        df = pd.concat([df,tmp], axis = 0)
    # sort the dataframe
    df = df.sort_values(by='CLARA_time_utc')
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # convert 1D array of floats to floats
        df[col] = df[col].apply(
            lambda x: float(x[0]) if (isinstance(x, (list, np.ndarray)) and np.array(x).ndim == 1 and len(x) == 1 and isinstance(x[0], (float, int, np.floating, np.integer)))
            else float(x) if isinstance(x, (float, int, np.floating, np.integer))
            else x)
        # convert coordinates array to 3 columns
        if 'eci' in col and len(df[col].iloc[0]) == 3:
            df[col + '_x'] = df[col].apply(lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df[col + '_y'] = df[col].apply(lambda x: float(x[1]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df[col + '_z'] = df[col].apply(lambda x: float(x[2]) if isinstance(x, (list, np.ndarray)) else np.nan)
            df = df.drop(col, axis = 1)
        # extract features from polygon
        elif isinstance(df[col].iloc[0], Polygon):
            df[col + '_area'] = df[col].apply(lambda x: x.area if isinstance(x, Polygon) else np.nan)
            df[col + '_length'] = df[col].apply(lambda x: x.length if isinstance(x, Polygon) else np.nan)
            df[col + '_centroid_x'] = df[col].apply(lambda x: x.centroid.x if isinstance(x, Polygon) else np.nan)
            df[col + '_centroid_y'] = df[col].apply(lambda x: x.centroid.y if isinstance(x, Polygon) else np.nan)
            df[col + '_vertices'] = df[col].apply(lambda x: len(x.exterior.coords) if isinstance(x, Polygon) else np.nan)
            df[col + '_bbox_minx'] = df[col].apply(lambda x: x.bounds[0] if isinstance(x, Polygon) else np.nan)
            df[col + '_bbox_miny'] = df[col].apply(lambda x: x.bounds[1] if isinstance(x, Polygon) else np.nan)
            df[col + '_bbox_maxx'] = df[col].apply(lambda x: x.bounds[2] if isinstance(x, Polygon) else np.nan)
            df[col + '_bbox_maxy'] = df[col].apply(lambda x: x.bounds[3] if isinstance(x, Polygon) else np.nan)
            df = df.drop(col, axis=1)
        # extract features from multipolygon
        elif isinstance(df[col].iloc[0], MultiPolygon):
            df[col + '_area'] = df[col].apply(lambda x: np.mean([p.area for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_length'] = df[col].apply(lambda x: np.mean([p.length for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_centroid_x'] = df[col].apply(lambda x: np.mean([p.centroid.x for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_centroid_y'] = df[col].apply(lambda x: np.mean([p.centroid.y for p in x.geoms]) if isinstance(x, MultiPolygon) else np.nan)
            df[col + '_vertices'] = df[col].apply(lambda x: int(np.mean([len(p.exterior.coords) for p in x.geoms])) if isinstance(x, Polygon) else np.nan)
            df[col + '_num_polygons'] = df[col].apply(lambda x: len(x.geoms) if isinstance(x, MultiPolygon) else np.nan)
            df = df.drop(col, axis = 1)
        else: 
            df[col + '_mean'] = df[col].apply(lambda x: pd.to_datetime(np.mean(pd.to_datetime(x).view(np.int64))) if isinstance(x, (list, np.ndarray)) else np.nan)
            df = df.drop(col, axis = 1)
    for col in df.columns:
        if 'CERES' in col:
            df = df.drop(col, axis=1)
    df.to_pickle(PATH_COMBINED)
else:
    df = pd.read_pickle(PATH_COMBINED)
features = df.columns
obj_cols = df.select_dtypes(include=['object']).columns
print('Number of features: ', len(features))
print('out of which ', len(obj_cols), ' are of type object.')
print(features)

