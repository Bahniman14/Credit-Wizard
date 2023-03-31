import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
from pydantic import BaseModel

class user_data(BaseModel):
     Loan_ID: str  
     Gender: str
     Married: str
     Dependents: int
     Education: str
     Self_Employed:str
     ApplicantIncome: int
     CoapplicantIncome:int
     LoanAmount:int
     Loan_Amount_Term: int
     Credit_History: int
     Property_Area:str

app = FastAPI()
# testdata = pd.read_csv('testCW.csv')
pickle_in = open("model.pkl", "rb")
naiveBayes = pickle.load(pickle_in)
   

def parseArray(obj):
     userData = [ obj.Loan_ID,obj.Gender,obj.Married,obj.Dependents,obj.Education,obj.Self_Employed,obj.ApplicantIncome,obj.CoapplicantIncome,obj.LoanAmount,obj.Loan_Amount_Term, obj.Credit_History,obj.Property_Area]
     # userData = pd.DataFrame(userData)
     return userData 

def modify(testdata):
        testdata['Gender'].fillna(testdata['Gender'].mode()[0], inplace=True)
        testdata['Dependents'].fillna(testdata['Dependents'].mode()[0], inplace=True)
        testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0], inplace=True)
        testdata.LoanAmount = testdata.LoanAmount.fillna((testdata.LoanAmount).mean())
     #    testdata.LoanAmount = str(testdata.LoanAmount.fillna((float(testdata.LoanAmount).mean())))
        testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0], inplace=True)
        testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0], inplace=True)
        testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])
        testdata['TotalIncome'] = testdata['ApplicantIncome'] + \
            testdata['CoapplicantIncome']
        testdata['TotalIncome_log'] = np.log(testdata['TotalIncome'])
        
        testdata["Dependents"]= testdata["Dependents"].map(str)
        testdata["Loan_Amount_Term"]= testdata["Loan_Amount_Term"].map(str)
        testdata["ApplicantIncome"]= testdata["Loan_Amount_Term"].map(str)
        testdata["CoapplicantIncome"]= testdata["CoapplicantIncome"].map(str)
        testdata["CoapplicantIncome"]= testdata["CoapplicantIncome"].map(str)
        testdata["LoanAmount"]= testdata["LoanAmount"].map(str)
        testdata["Loan_Amount_Term"]= testdata["Loan_Amount_Term"].map(str)
        testdata["Credit_History"]= testdata["Credit_History"].map(str)
        testdata["TotalIncome"]= testdata["TotalIncome"].map(str)
        testdata["TotalIncome_log"]= testdata["TotalIncome_log"].map(str)


        test = testdata.iloc[:, np.r_[1:5, 9:11, 13:15]].values
        SS = StandardScaler()
        LE_test = LabelEncoder()
        for i in range(0, 5):
             test[:, i] = LE_test.fit_transform(test[:, i])
        test[:, 7] = LE_test.fit_transform(test[:, 7])
        test = SS.fit_transform(test)
        return test




@app.get('/')
def index():
    return {'message': 'Hello!!'}
 
# under development:

@app.post('/predict')
async def get(obj: user_data):
     testdata = pd.read_csv('testCW.csv')
     userData = parseArray(obj)
     print(len(testdata))
     # print(type(userData))

     testdata = testdata.append(pd.Series(userData, index=testdata.columns), ignore_index=True)
     # print(len(testdata))
     test = modify(testdata)
     # print(test)
     # print(len(test))
     prediction = naiveBayes.predict(test)
     # print(prediction)
     # print(len(prediction))
     return {f'{prediction[len(prediction) - 1]}'}
     # return 1


if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1', port=8000)

    # uvicorn main:app --reload

# how it will work
# 1. The chat bot will collect the data and upload it in test data in the database
# 1.5 The DataBase file will be feed to the model by the fast api (main) file
# 2.the Model will fill all null values
# then it will make prediction and will print: prediction[367]
# then fast api will return this prediction value which will be read by the chat bot and it will respond to the user acording to it
#
#Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
#   when it is 1
#   "Loan_ID": "LP002990",
#   "Gender": "Male",
#   "Married": "No",
#   "Dependents": "2",
#   "Education": "Graduate",
#   "Self_Employed": "Yes",
#   "ApplicantIncome": "9200",
#   "CoapplicantIncome": "1000",
#   "LoanAmount": "98",
#   "Loan_Amount_Term": "180",
#   "Credit_History": "1",
#   "Property_Area": "Rural"

# LP001056,Male,Yes,2,Not Graduate,No,3881,0,147,360,0,Rural
# when it is 0
#   "Loan_ID": "LP001056",
#   "Gender": "Male",
#   "Married": "Yes",
#   "Dependents": "2",
#   "Education": "Not Graduate",
#   "Self_Employed": "No",
#   "ApplicantIncome": "3881",
#   "CoapplicantIncome": "0",
#   "LoanAmount": "147",
#   "Loan_Amount_Term": "360",
#   "Credit_History": "0",
#   "Property_Area": "Rural"

# The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
