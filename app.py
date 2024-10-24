import streamlit as st
import pickle 
import pandas as pd
import numpy as np
import xgboost as xgb
columns = pickle.load(open('columns.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
education_encode = pickle.load(open('education_encode.pkl','rb'))
dummy_col = pickle.load(open('dummy_column.pkl','rb'))

# st.title('Welcome to CREDIT FRAUD DETECTION')

for i in columns:
    print(i)

