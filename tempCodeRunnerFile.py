testdata = pd.read_csv('testCW.csv')
# ** cheaking there is any missing value or not **
# print(testdata.isnull().sum())
testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace=True)
testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace=True)
testdata['Self_Employed'].fillna(
    testdata['Self_Employed'].mode()[0], inplace=True)
testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())
testdata['Loan_Amount_Term'].fillna(
    testdata['Loan_Amount_Term'].mode()[0], inplace=True)
testdata['Credit_History'].fillna(
    testdata['Credit_History'].mode()[0], inplace=True)
testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])
# print(testdata.isnull().sum())

testdata['TotalIncome'] = testdata['ApplicantIncome'] + \
    testdata['CoapplicantIncome']
testdata['TotalIncome_log'] = np.log(testdata['TotalIncome'])
# print(testdata)
test = testdata.iloc[:, np.r_[1:5, 9:11, 13:15]].values
LE_test = LabelEncoder()
for i in range(0, 5):
    test[:, i] = LE_test.fit_transform(test[:, i])
test[:, 7] = LE_test.fit_transform(test[:, 7])
# print(test)
test = SS.fit_transform(test)
# print(test)
prediction = naiveBayes.predict(test)
# print(prediction)
print(prediction[366])