import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %mattplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pickle

dataset = pd.read_csv('trainCW.csv')
# print(dataset.head)
# **Normalizing LoanAmount_log**
dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])

# **Finding Missing Values present in each of the variables in the data set **
# print(dataset.isnull().sum())
# **Filing the Missing values with the mode value of other existing values in that column **
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace=True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace=True)
dataset['Self_Employed'].fillna(
    dataset['Self_Employed'].mode()[0], inplace=True)
dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(
    dataset.LoanAmount_log.mean())
dataset['Loan_Amount_Term'].fillna(
    dataset['Loan_Amount_Term'].mode()[0], inplace=True)
dataset['Credit_History'].fillna(
    dataset['Credit_History'].mode()[0], inplace=True)
# print(dataset.isnull().sum())

# **Normalizing Other Missing Datas
dataset['TotalIncome'] = dataset['ApplicantIncome'] + \
    dataset['CoapplicantIncome']
dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])

# **Normalization and filling up missing data is done!
# print(dataset.head())
# **Storing Dependent and Indepented Variabes saparately
x = dataset.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = dataset.iloc[:, 12].values
# print("[INDIPENDENT VARIABLES(x)]:")
# print(x)
# print("[DIPENDENT VARIABLES(Y)]:")
# print(y)

# **Spliting the dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
# print("X_train:")
# print(x_train)
# **********Encoding the Data:
LE_X = LabelEncoder()
for i in range(0, 5):
    x_train[:, i] = LE_X.fit_transform(x_train[:, i])
x_train[:, 7] = LE_X.fit_transform(x_train[:, 7])
# print("X_train(Lable_Encoded):")
# print(x_train)

LE_Y = LabelEncoder()
y_train = LE_Y.fit_transform(y_train)
# print("Y_train(Lable_Encoded):")
# print(y_train)

for i in range(0, 5):
    x_test[:, i] = LE_X.fit_transform(x_test[:, i])
x_test[:, 7] = LE_X.fit_transform(x_test[:, 7])
# print("x_test(Lable_Encoded):")
# print(x_test)

y_test = LE_Y.fit_transform(y_test)
# print("Y_test(Lable_Encoded):")
# print(y_test)

# *Scaling the data set
SS = StandardScaler()
x_train = SS.fit_transform(x_train)
x_test = SS.fit_transform(x_test)

# Appling Decision Tree Algorithm (Uneffective)
# decisiontree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# decisiontree.fit(x_train, y_train)
# y_pred = decisiontree.predict(x_test)
# print("prediction:")
# print(y_pred)
# print("Decision Tree Prediction Accuracy: ", metrics.accuracy_score(y_pred, y_test))

# Appling Naive Bayes Algorithm (effective)
naiveBayes = GaussianNB()
naiveBayes.fit(x_train, y_train)

y_pred = naiveBayes.predict(x_test)
# print("prediction:")
# print(y_pred)
print("Decision Tree Prediction Accuracy: ",
      metrics.accuracy_score(y_pred, y_test))

# Reading Test Data:
# testdata = pd.read_csv('testCW.csv')
# # ** cheaking there is any missing value or not **
# # print(testdata.isnull().sum())
# testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace=True)
# testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace=True)
# testdata['Self_Employed'].fillna(
#     testdata['Self_Employed'].mode()[0], inplace=True)
# testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())
# testdata['Loan_Amount_Term'].fillna(
#     testdata['Loan_Amount_Term'].mode()[0], inplace=True)
# testdata['Credit_History'].fillna(
#     testdata['Credit_History'].mode()[0], inplace=True)
# testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])
# # print(testdata.isnull().sum())

# testdata['TotalIncome'] = testdata['ApplicantIncome'] + \
#     testdata['CoapplicantIncome']
# testdata['TotalIncome_log'] = np.log(testdata['TotalIncome'])
# # print(testdata)
# test = testdata.iloc[:, np.r_[1:5, 9:11, 13:15]].values
# LE_test = LabelEncoder()
# for i in range(0, 5):
#     test[:, i] = LE_test.fit_transform(test[:, i])
# test[:, 7] = LE_test.fit_transform(test[:, 7])
# # print(test)
# test = SS.fit_transform(test)
# # print(test)
# prediction = naiveBayes.predict(test)
# # print(prediction)
# print(prediction[366])

pickle_out = open("model.pkl", "wb")
pickle.dump(naiveBayes, pickle_out)
pickle_out.close()
