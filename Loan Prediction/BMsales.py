#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#reading dataset
dataset = pd.read_csv('train.csv')

#taking care of missing data
dataset.isnull().sum()
#1 weight
dataset['Item_Weight'].fillna(dataset['Item_Weight'].mean(),inplace=True)
#2 outlet size
dataset['Outlet_Size'].value_counts()
dataset['Outlet_Size'].fillna('Medium',inplace=True)

#encoding categorical variables
from sklearn.preprocessing import LabelEncoder
""" This function is universal to onehot encode the whole data-set"""

#Auto encodes any dataframe column of type category or object.
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df

dummyEncode(dataset)

#spliting into X & y
X=dataset.iloc[:,[1,2,3,4,5,6,8,9,10]].values
y=dataset.iloc[:,11].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X_test)
