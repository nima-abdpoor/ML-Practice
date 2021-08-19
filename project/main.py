import pandas as pd

melbournFilePath = 'D:/nima/python/melb_data.csv'
melbournData = pd.read_csv(melbournFilePath)
melbourn_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


def drop_not_available_data():
    melbournData.dropna(axis=0)


def print_all_columns():
    print(melbournData.columns)


def get_price():
    return melbournData.Price


def getDataByFeature(feature: list):
    return melbournData[feature]


def getDataByFeatureDescribe(feature: list):
    return melbournData[feature].describe()


def getDataByFeatureWithHead(feature: list, number: int = 7):
    return melbournData[feature].head(number)


print(getDataByFeatureWithHead(melbourn_features,25))
