import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
def modify(test):
        testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace=True)
        testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace=True)
        testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace=True)
        testdata.LoanAmount = testdata.LoanAmount.fillna(testdata.LoanAmount.mean())
        testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0], inplace=True)
        testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0], inplace=True)
        testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])
        testdata['TotalIncome'] = testdata['ApplicantIncome'] + \
            testdata['CoapplicantIncome']
        testdata['TotalIncome_log'] = np.log(testdata['TotalIncome'])
        test = testdata.iloc[:, np.r_[1:5, 9:11, 13:15]].values
        SS = StandardScaler()
        LE_test = LabelEncoder()
        for i in range(0, 5):
             test[:, i] = LE_test.fit_transform(test[:, i])
        test[:, 7] = LE_test.fit_transform(test[:, 7])
        test = SS.fit_transform(test)
        return test

app = FastAPI()
testdata = pd.read_csv('testCW.csv')
pickle_in = open("model.pkl", "rb")
naiveBayes = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Hello!!'}

# under development:
@app.get('/predict')
def predict():
     test = modify(testdata)
     prediction = naiveBayes.predict(test)
    #  return prediction[366]
     return {f'{prediction[7]}'}

if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)

    # uvicorn main:app --reload

# how it will work
# 1. The chat bot will collect the data and upload it in test data in the database
# 1.5 The DataBase file will be feed to the model by the fast api (main) file
# 2.the Model will fill all null values
# then it will make prediction and will print: prediction[367]
# then fast api will return this prediction value which will be read by the chat bot and it will respond to the user acording to it