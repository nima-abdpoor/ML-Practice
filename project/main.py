import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbournFilePath = 'D:/nima/python/melb_data.csv'
melbournData = pd.read_csv(melbournFilePath)
melbourn_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
melbourn_featuresTwo = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'Price']


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


def predict():
    melbournModel = DecisionTreeRegressor(random_state=1)
    print(getDataByFeatureWithHead(melbourn_features), '\n***\n')
    print(getDataByFeatureWithHead(melbourn_featuresTwo, 100), '\n***\n')
    print(get_price(), '\n***\n')
    melbournModel.fit(getDataByFeature(melbourn_features), get_price())
    print("Making predictions for the following 7 houses:\n")
    print(melbournModel.predict(getDataByFeatureWithHead(melbourn_features)))


print(predict())
