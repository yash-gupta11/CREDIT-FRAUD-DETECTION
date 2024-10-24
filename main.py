# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:06:17 2024

@author: gyash
"""
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import seaborn as sns
import pickle



a1 = pd.read_excel(r"C:\Users\gyash\Desktop\campusX\PROJECT\CREDIT FRAUD DETECTION\case_study1.xlsx")
a2 = pd.read_excel(r"C:\Users\gyash\Desktop\campusX\PROJECT\CREDIT FRAUD DETECTION\case_study2.xlsx")

df1 = a1.copy()
df2 = a2.copy()


# handling null values(removing null values)

df1 = df1[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []
#selecting columns to be drop as null values is more than 10000(20%)
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)

df2.drop(columns = columns_to_be_removed,inplace = True)

for i in df2.columns:
    df2 = df2[df2[i] != -99999]
    
for i in df1.columns:
    if i in df2.columns:
        common_col = i

df = pd.merge(df1,df2, how='inner',on=common_col)

numerical_col = []
categorical_col = []
target_col = []


for i in df.columns:
    if df[i].dtype == 'object':
        categorical_col.append(i)
    
for i in df.columns:
    if df[i].dtype in ['int64','float64']:
        numerical_col.append(i)

target_col.append('Approved_Flag')    
categorical_col.remove('Approved_Flag')
numerical_col.remove('PROSPECTID')
# feature selection
#chi sqaure test
for i in categorical_col:
    stat, p, dof, expected = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i,'===',p)
#since p value of each column is less then 0.05 then we have to keep all the columns    


# multicollinearity


vif_data = df[numerical_col]
column_index= 0
numerical_col_to_be_kept = []
for i in range(0,len(numerical_col)):
    vif = variance_inflation_factor(vif_data,column_index)
    if vif <= 6:
        numerical_col_to_be_kept.append(numerical_col[i])
        column_index = column_index+1
    else:
        vif_data.drop(numerical_col[i],axis = 1,inplace=True)

df = df[categorical_col+numerical_col_to_be_kept + target_col]

# ANOVA
final_numerical_col = []
for i in numerical_col_to_be_kept:
    group1 = df.loc[df['Approved_Flag'] == 'P1'][i].values
    group2 = df.loc[df['Approved_Flag'] == 'P2'][i].values
    group3 = df.loc[df['Approved_Flag'] == 'P3'][i].values
    group4 = df.loc[df['Approved_Flag'] == 'P4'][i].values
    stat,p = f_oneway(group1,group2,group3,group4)
    print(i,'===',p)
    if p <= 0.05 :
        final_numerical_col.append(i)
        
        
df = df[final_numerical_col+categorical_col+['Approved_Flag']]
        
#Data Encoding

# ORDINAL ENCODING 

# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


education_encode = {
        'SSC'            : 1,
        'OTHERS'         : 1,
        '12TH'           : 2,
        'GRADUATE'       : 3,
        'UNDER GRADUATE' : 3,
        'POST-GRADUATE'  : 4,
        'PROFESSIONAL'   : 5
    
    }

df['EDUCATION'] = df['EDUCATION'].replace(education_encode)

# ONE HOT ENCODING

encoded_df = pd.get_dummies(data = df,columns = ['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'],dtype=np.int8)


# Model training(Seletion)

# 01.Random Forest     
# 02.Decision Tree Classifier  
# 03.XGBoost

# Out of these three model best accuracy is in XGBoost
# So after the Hyperparameter Tuning the best model is given below


x,y = encoded_df.drop(columns = 'Approved_Flag'),encoded_df['Approved_Flag']

le = LabelEncoder()
y = le.fit_transform(y)


# Best Hyperparameter ====== {'learning_rate': [0.2], 'max_depth': [3], 'n_estimators': [200],}
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
model = xgb.XGBClassifier(learning_rate = 0.2, max_depth = 3, n_estimators=200,objective='multi:softmax',  num_class=4)
model.fit(x_train,y_train)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)
print("test accuracy : ",accuracy_score(y_test,test_pred))
print("train accuracy",accuracy_score(y_train,train_pred))
pickle.dump(model,open('model.pkl','wb')) 
pickle.dump(x_train.columns,open('dummy_column.pkl','wb'))
pickle.dump(education_encode,open('education_encode.pkl','wb'))
col = final_numerical_col+categorical_col+['Approved_Flag']
pickle.dump(col,open('columns.pkl','wb'))
unseen_df = pd.read_excel(r"C:\Users\gyash\Desktop\campusX\PROJECT\CREDIT FRAUD DETECTION\Unseen_Dataset.xlsx")

unseen_df = pd.get_dummies(data = unseen_df,columns = ['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'],dtype=np.int8)

unseen_df['EDUCATION'] = unseen_df['EDUCATION'].replace(education_encode)

final_pred = model.predict(unseen_df[x_train.columns])

values,counts = np.unique(final_pred, return_counts=True)
for value,count in zip(values,counts):
    print(value,"....",count,"....",count/len(final_pred))

values,counts = np.unique(y_train, return_counts=True)    
for value,count in zip(values,counts):
    print(value,"....",count,"....",count/y_train.shape[0])

values,counts = np.unique(y, return_counts=True)    
for value,count in zip(values,counts):
    print(value,"....",count,"....",count/y_train.shape[0])












