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
# from pydantic import BaseModel

class user_data():
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
     Credit_History:int
     Property_Area:str

testdata = pd.read_csv('testCW.csv')

# user_data = {,Male,No,0,Graduate,Yes,9200,0,98,180,1,Rural}

C = user_data()
C.Loan_ID = "LP002990"
C.Gender = "Male"
C.Married = "No"
C.Dependents = 0
C.Education = "Graduate"
C.Self_Employed = "Yes"
C.ApplicantIncome = 9200
C.CoapplicantIncome = 0
C.LoanAmount = 98
C.Loan_Amount_Term = 180
C.Credit_History = 1
C.Property_Area = "Urban"
S =  [C.Loan_ID,C.Gender,C.Married,C.Dependents,C.Education,C.Self_Employed,C.ApplicantIncome,C.CoapplicantIncome,C.LoanAmount,C.Loan_Amount_Term, C.Credit_History,C.Property_Area]
S = pd.DataFrame(S)
# 
print(type(testdata))
print(type(S))
print(len(testdata))
# testdata.merge(S)
print(len(testdata))
