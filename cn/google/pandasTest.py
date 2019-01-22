import pandas as pd
import tensorflow  as tf
print(pd.__version__)

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
t = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(t)


#california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = pd.read_csv("C:\\Users\\wm\Downloads\\california_housing_train.csv", sep=",")
#california_housing_dataframe.describe()
#california_housing_dataframe.head()
#california_housing_dataframe.hist('housing_median_age')
print(california_housing_dataframe.head())

"""
http://pandas.pydata.org/pandas-docs/stable/indexing.html
"""