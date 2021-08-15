import pandas as pd

melbournFilePath = 'D:/nima/python/melb_data.csv'
melbournData = pd.read_csv(melbournFilePath)


print(melbournData.describe())