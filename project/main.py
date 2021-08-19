import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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
    melbournModel.fit(getDataByFeature(melbourn_features), get_price())
    return melbournModel.predict(getDataByFeature(melbourn_features))


def calculateValidation():
    predictedHomePrices = predict()
    return mean_absolute_error(get_price(), predictedHomePrices)


def calculateValidationWithDataSepration():
    trainX, valX, trainY, valY = train_test_split(
        getDataByFeature(melbourn_features),
        get_price(),
        random_state=0)
    melbournModel = DecisionTreeRegressor()
    melbournModel.fit(trainX, trainY)
    prediction = melbournModel.predict(valX)
    return mean_absolute_error(valY, prediction)


print(calculateValidationWithDataSepration())
print(calculateValidation())
