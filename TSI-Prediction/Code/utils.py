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