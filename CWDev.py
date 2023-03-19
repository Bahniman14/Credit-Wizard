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

dataset = pd.read_csv('trainCW.csv')
print(dataset.head)
# print(dataset.shape)
# print(pd.crosstab(dataset['Credit_History'], dataset['Loan_Status'], margins= True))
# dataset.boxplot(column='ApplicantIncome')
# plt.show()
# dataset['ApplicantIncome'].hist(bins=20)
# plt.show()
# dataset['CoapplicantIncome'].hist(bins=20)
# plt.show()
# dataset['LoanAmount'].hist(bins=20)
# plt.show()
# **Normalizing LoanAmount_log**
dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
# dataset['LoanAmount_log'].hist(bins=20)
# plt.show()

# **Finding Missing Values present in each of the variables in the data set **
print(dataset.isnull().sum())
# **Filing the Missing values with the mode value of other existing values in that column **
dataset['Gender'].fillna(dataset['Gender'].mode()[0], inplace= True)
dataset['Married'].fillna(dataset['Married'].mode()[0], inplace= True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0], inplace= True)
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0], inplace= True)
dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0], inplace= True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace= True)
print(dataset.isnull().sum())
# **Normalizing Other Missing Datas
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome_log'] = np.log(dataset['TotalIncome'])
# dataset['TotalIncome'].hist(bins= 20)
# plt.show()
# dataset['TotalIncome_log'].hist(bins= 20)
# plt.show()
# **Normalization and filling up missing data is done!
print(dataset.head())
# **Storing Dependent and Indepented Variabes saparately
x = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y = dataset.iloc[:,12].values
print("[INDIPENDENT VARIABLES(x)]:")
print(x)
print("[DIPENDENT VARIABLES(Y)]:")
print(y)

# **Spliting the dataset into train and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("X_train:")
print(x_train)
# **********Encoding the Data:
LE_X = LabelEncoder()
for i in range(0,5):
        x_train[:,i] = LE_X.fit_transform(x_train[:,i])
x_train[:,7] = LE_X.fit_transform(x_train[:,7])
print("X_train(Lable_Encoded):")
print(x_train)

LE_Y = LabelEncoder()
y_train = LE_Y.fit_transform(y_train)
print("Y_train(Lable_Encoded):")
print(y_train)

for i in range(0,5):
        x_test[:,i] = LE_X.fit_transform(x_test[:,i])
x_test[:,7] = LE_X.fit_transform(x_test[:,7])
print("x_test(Lable_Encoded):")
print(x_test)

y_test = LE_Y.fit_transform(y_test)
print("Y_test(Lable_Encoded):")
print(y_test)

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
print("prediction:")
print(y_pred)
print("Decision Tree Prediction Accuracy: ", metrics.accuracy_score(y_pred, y_test))

testdata_1 = pd.read_csv('testCW1.csv')
testdata_1['LoanAmount_log'] = np.log(testdata_1['LoanAmount'])
# ** cheaking there is any missing value or not **
print(testdata_1.isnull().sum())
testdata_1['TotalIncome'] = testdata_1['ApplicantIncome'] + testdata_1['CoapplicantIncome']
testdata_1['TotalIncome_log'] = np.log(testdata_1['TotalIncome'])
test_1 = testdata_1.iloc[:,np.r_[1:5,9:11,13:15]].values
LE_test_1 = LabelEncoder()
for i in range(0,5):
        test_1[:,i] = LE_test_1.fit_transform(test_1[:,i])
test_1[:,7] = LE_test_1.fit_transform(test_1[:,7])
test_1 = SS.fit_transform(test_1)
print(test_1)
predtest_1 = naiveBayes.predict(test_1)


testdata_2 = pd.read_csv('testCW2.csv')
testdata_2['LoanAmount_log'] = np.log(testdata_1['LoanAmount'])
# ** cheaking there is any missing value or not **
print(testdata_2.isnull().sum())
testdata_2['TotalIncome'] = testdata_1['ApplicantIncome'] + testdata_1['CoapplicantIncome']
testdata_2['TotalIncome_log'] = np.log(testdata_1['TotalIncome'])
test_2 = testdata_2.iloc[:,np.r_[1:5,9:11,13:15]].values
LE_test_2 = LabelEncoder()
for i in range(0,5):
        test_2[:,i] = LE_test_2.fit_transform(test_1[:,i])
test_2[:,7] = LE_test_1.fit_transform(test_1[:,7])
test_2 = SS.fit_transform(test_1)
print(test_2)
predtest_2 = naiveBayes.predict(test_2)


testdata_8 = pd.read_csv('testCW8.csv')
testdata_8['LoanAmount_log'] = np.log(testdata_8['LoanAmount'])
# ** cheaking there is any missing value or not **
print(testdata_8.isnull().sum())
testdata_8['TotalIncome'] = testdata_8['ApplicantIncome'] + testdata_8['CoapplicantIncome']
testdata_8['TotalIncome_log'] = np.log(testdata_8['TotalIncome'])
test_8 = testdata_8.iloc[:,np.r_[1:5,9:11,13:15]].values
LE_test_8 = LabelEncoder()
for i in range(0,5):
        test_8[:,i] = LE_test_8.fit_transform(test_8[:,i])
test_8[:,7] = LE_test_8.fit_transform(test_8[:,7])
test_8 = SS.fit_transform(test_8)
print(test_8)
predtest_8 = naiveBayes.predict(test_8)





print(f"Prediction on test data.1 : {predtest_1}")
print(f"Prediction on test data.1 : {predtest_2}")
print(f"Prediction on test data.1 : {predtest_8}")