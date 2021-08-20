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


def get_MAE(max_leaf_nodes, trainX, trainY, valX, valY):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(trainX, trainY)
    predictionVal = model.predict(valX)
    mae = mean_absolute_error(valY, predictionVal)
    return mae


# compare MAE with differing values of max_leaf_nodes
def calculateMAEWithLeafNodes():
    trainX, valX, trainY, valY = train_test_split(
        getDataByFeature(melbourn_features),
        get_price(),
        random_state=0)
    maeResults = []
    for max_leaf_nodes in [50, 500, 5000, 50000]:
        my_mea = get_MAE(max_leaf_nodes, trainX, trainY, valX, valY)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mea))
        maeResults.append(my_mea.__int__())
    return maeResults


print(calculateMAEWithLeafNodes().__getitem__(2))
