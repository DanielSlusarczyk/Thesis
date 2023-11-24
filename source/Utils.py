import numpy as np
import pandas as pd

def DescribeData(df: pd.DataFrame):
    nmbOfRows = df.shape[0]
    nmbOfColumns = df.shape[1]
    

    print(f'Size: {nmbOfRows} x {nmbOfColumns}\n')
    print(pd.DataFrame(df.isnull().sum(), columns=['Number of Nans']).T)
    print(df.describe())

    print(df.head(3))

def SplitDateColumn(df: pd.DataFrame, column: str, suffix = '', replace=False):
    
    if suffix == '':
        suffix = column

    df[suffix + '_minute'] = df[column].dt.minute
    df[suffix + '_hour'] = df[column].dt.hour
    df[suffix + '_day'] = df[column].dt.day
    df[suffix + '_month'] = df[column].dt.month
    df[suffix + '_year'] = df[column].dt.year
    df[suffix + '_datetime'] = df[column]
    df[suffix + '_time'] = df[column].dt.time
    df[suffix + '_date'] = df[column].dt.date

    if replace:
        df.drop(column, axis = 1, inplace=True)

    return df