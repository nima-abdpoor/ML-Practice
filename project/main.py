import pandas as pd

melbournFilePath = 'D:/nima/python/melb_data.csv'
melbournData = pd.read_csv(melbournFilePath)


def drop_not_available_data():
    melbournData.dropna(axis=0)


def print_all_columns():
    print(melbournData.columns)


def get_price():
    return melbournData.Price


print(get_price())
