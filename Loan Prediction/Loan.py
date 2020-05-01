# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset 
dataset = pd.read_csv('train_ctrUa4K.csv')

#taking care of missing values
dataset.isnull().sum()
  #1 Gender
  dataset['Gender'].value_counts()
  dataset['Gender'].fillna("Male", inplace=True)
  #2 Married
  dataset['Married'].value_counts()
  dataset['Married'].fillna("Yes", inplace=True)
  #3 Dependents
  dataset['Dependents'].value_counts()
  dataset['Dependents'].fillna("0", inplace=True)
  dataset['Dependents']=  dataset['Dependents'].apply(str)

  #4 Self Employed
  dataset['Self_Employed'].value_counts()
  dataset['Self_Employed'].fillna("No", inplace=True)
  #5 loan amt
  dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace= True)
  #6 loan amt term
  dataset['Loan_Amount_Term'].value_counts()
  dataset['Loan_Amount_Term'].fillna("360", inplace=True)
  dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].apply(int)
  dataset['Loan_Amount_Term']=dataset['Loan_Amount_Term'].apply(str)

  #7 Credit History
  dataset['Credit_History'].value_counts()
  dataset['Credit_History'].fillna("1", inplace=True)
  dataset['Credit_History']=dataset['Credit_History'].apply(int)
  dataset['Credit_History']=dataset['Credit_History'].apply(str)


#splitting dataset into X and y
  X= dataset.iloc[:, 1:12].values
  y=dataset.iloc[:,12].values
  
# Encoding categorical data
  # Gender
  from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  labelencoder_X = LabelEncoder()
  X[:,0] = labelencoder_X.fit_transform(X[:,0])

  # Married
  from sklearn.preprocessing import LabelEncoder, OneHotEncoder
  labelencoder_X = LabelEncoder()
  X[:,1] = labelencoder_X.fit_transform(X[:,1])

  # dependents
  X[:,2] = labelencoder_X.fit_transform(X[:,2])
  onehotencoder = OneHotEncoder(categorical_features = [2])

  #education
  X[:,3] = labelencoder_X.fit_transform(X[:,3])
  # Self Employed
  X[:,4] = labelencoder_X.fit_transform(X[:,4])
  #loan amt term
  X[:,8] = labelencoder_X.fit_transform(X[:,8])
  onehotencoder = OneHotEncoder(categorical_features = [8])
  

  #Credit History
  X[:,9] = labelencoder_X.fit_transform(X[:,9])
  #Property Area
  X[:,10] = labelencoder_X.fit_transform(X[:,10])
  onehotencoder = OneHotEncoder(categorical_features = [10])
  
X = onehotencoder.fit_transform(X).toarray()
X=np.asarray(X)
X = X[X.columns[0:-1]]
print(X.columns)

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
np.arange(X).tolist()

with np.printoptions(threshold=numpy.inf):
    print(X)
  