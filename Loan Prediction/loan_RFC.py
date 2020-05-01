# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset 
dataset = pd.read_csv('train_ctrUa4K.csv')


#print(dataset)
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


################################################################################################################
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
##################################################################################################################

#splitting dataset into X and y
X= dataset.iloc[:, 1:12].values
#convert x and y to numpy arrays
X = np.asarray(X)
y=dataset.iloc[:,12].values
y = np.asarray(y)
print('variables:', X)
X = dataset[dataset.columns[1:-1]]

#splitting dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#fitting random forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
#predicting
y_pred= classifier.predict(X_test)

#making confusion matrix
 from sklearn.metrics import confusion_matrix
 cm=confusion_matrix(y_test,y_pred)
 acc=(100+25)/154