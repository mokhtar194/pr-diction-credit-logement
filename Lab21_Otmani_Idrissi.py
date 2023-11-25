# Lab21 : Prédiction Credit Logement
# Réalisé par Otmani Idrissi Mokhtar EMSI 2023/2024
# Reference

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from io import StringIO

# Step 1 : DATASET
dt = pd.read_csv("datasets/train.csv")
print(dt.head())
print(dt.info())
print(dt.isna().sum())

# Data Transformation
# """
# Remplacement des valeures manquantes
# Nous allons remplacer les variables manquantes categoriques par leurs modes
# Nous allons remplacer les variables manquantes numériques par la médiane
# """


def trans(data):
    for c in data.columns:
        if data[c].dtype == 'int64' or data[c].dtype == 'float64':
            data[c].fillna(data[c].median(), inplace=True)
        else:
            data[c].fillna(data[c].mode()[0], inplace=True)


trans(dt)


print(dt.isna().sum())

var_num = dt[["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]]
print(var_num.describe())

var_cat = dt[
    ["Loan_Status", "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Credit_History"]]
var_cat = pd.get_dummies(var_cat, drop_first=True)
print(var_cat.head())

transformed_dataset = pd.concat([var_cat, var_num], axis=1)
transformed_dataset.to_csv("datasets/transformed_dataset.csv")

# Split DataSet on target y and features x
y = transformed_dataset["Loan_Status_Y"]
x = transformed_dataset.drop("Loan_Status_Y", axis=1)

# Train 80% Test 20% Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# Data Visualisation
# print(dt["Loan_Status"].value_counts())
# print(dt["Loan_Status"].value_counts(normalize=True) * 100)
#
# fig = px.histogram(dt, x="Loan_Status", title='Crédit accordé ou pas', color="Loan_Status", template='plotly_dark')
# fig.show(font=dict(size=17, family="Franklin Gothic"))
#
# fig = px.pie(dt, names="Dependents", title='Dependents', color="Dependents", template='plotly_dark')
# fig.show(font=dict(size=17, family="Franklin Gothic"))

# Step 2 : MODEL
# model = SVC() | SGDClassifier() | SGDClassifier 77
# model = RandomForestClassifier() | LogisticRegression 85
# model = AdaBoostClassifier() 80
# model = GradientBoostingClassifier() 81
model = LogisticRegression()

# Step 3 : TRAIN
model.fit(x_train, y_train)

# Step 4 : TEST
print("Model Accuracy :", model.score(x_test, y_test) * 100, "%")

# Web Deployment Of The Model : streamlit run filename.py
st.header('Prévision Crédit Logement')
st.subheader('Enter Client Data')


def data_struct(inputD):
    trans(inputD)
    data_result = pd.DataFrame(columns=transformed_dataset.columns)
    data_result = data_result.drop("Loan_Status_Y", axis=1)
    for index, row in inputD.iterrows():
        data = {
            'Credit_History': row["Credit_History"],
            'Gender_Male': True if row["Gender"] == 'Male' else False,
            'Married_Yes': True if row["Married"] == 'Yes' else False,
            # 'Dependents_0': True if row["Dependents"] == '0' else False,
            'Dependents_1': True if row["Dependents"] == '1' else False,
            'Dependents_2': True if row["Dependents"] == '2' else False,
            'Dependents_3+': True if row["Dependents"] == '3+' else False,
            'Education_Not Graduate': True if row["Education"] == 'Not Graduate' else False,
            'Self_Employed_Yes': True if row["Self_Employed"] == 'Yes' else False,
            'Property_Area_Semiurban': True if row["Property_Area"] == 'Semiurban' else False,
            'Property_Area_Urban': True if row["Property_Area"] == 'Urban' else False,
            'ApplicantIncome': row["ApplicantIncome"],
            'CoapplicantIncome': row["CoapplicantIncome"],
            'LoanAmount': row["LoanAmount"],
            'Loan_Amount_Term': row["Loan_Amount_Term"]
        }
        df2 = pd.DataFrame([data])
        # print('df2', df2)
        data_result = pd.concat([data_result, df2]) if not data_result.empty else df2
        # print('data res', data_result)
    return data_result


def single_userInfos():
    col1, col2, col21 = st.columns(3)
    with col1:
        loanid = st.text_input('Loan_Id', placeholder='enter client Loan Id')
        gender = st.selectbox(
            'Gender',
            ('Female', 'Male'))
    with col2:
        graduation = st.selectbox(
            'Education',
            ('Graduate', 'Not Graduate'))
    with col21:
        propertyArea = st.selectbox(
            'Property Area',
            ('Semi-urban', 'Urban', 'Rural'))
    col3, col4 = st.columns(2)
    with col3:
        dependency = st.selectbox(
            'Dependents',
            ('0', '1', '2', '3+'))
    with col4:
        married = st.selectbox(
            'Married',
            ('Yes', 'No'))
    col5, col6 = st.columns(2)
    with col5:
        selfEmployed = st.selectbox(
            'Self Employed',
            ('Yes', 'No'))
    with col6:
        creditHistory = st.number_input('Client Credit History')
    col7, col8 = st.columns(2)
    with col7:
        applicantIncome = st.number_input('Client Applicant Income')
    with col8:
        coapplicantIncome = st.number_input('Client Coapplicant Income')
    col9, col10 = st.columns(2)
    with col9:
        loanAmount = st.number_input('Client Loan amount')
    with col10:
        loanTerm = st.number_input('Client Loan Term')

    data = {
        'Loan_ID': loanid,
        'Credit_History': creditHistory,
        'Gender': gender,
        'Married': married,
        'Education': graduation,
        'Self_Employed': selfEmployed,
        'Property_Area': propertyArea,
        'Dependents': dependency,
        'ApplicantIncome': applicantIncome,
        'CoapplicantIncome': coapplicantIncome,
        'LoanAmount': loanAmount,
        'Loan_Amount_Term': loanTerm,
    }
    client_features = pd.DataFrame([data])
    transformedData = data_struct(client_features)
    if client_features.empty:
        return client_features, client_features["Loan_ID"]
    return transformedData, client_features["Loan_ID"]


df, loan_id = single_userInfos()
st.write(df)
# print(df)

prediction = model.predict(df)
print(prediction)

if st.button('Predict'):
    st.write(prediction)


st.sidebar.header('Crédit Logement Features')
st.sidebar.subheader('Choose The file containing Client folders to treat')
uploaded_file = st.sidebar.file_uploader("Drag your file here")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)
    if not dataframe.empty:
        dfg = data_struct(dataframe)

if st.sidebar.button('Predict Group'):
    # st.write('prediction in cour')
    if dfg is not None:
        dict_line = {}
        list_col = []
        i = 0
        for index, row in dfg.iterrows():
            prediction = model.predict([row])

            dict_line = {
                'Loan_ID': dataframe.iloc[i]['Loan_ID'],
                'Prediction': prediction
            }
            list_col.append(dict_line)
            i = i + 1
            if len(list_col) == 4:
                colo1, colo2, colo3, colo4 = st.columns(4)
                with colo1:
                    st.write("Loan ID :  ", list_col[0]['Loan_ID'])
                    st.write("Prediction :  ", list_col[0]['Prediction'])
                with colo2:
                    st.write("Loan ID :  ", list_col[1]['Loan_ID'])
                    st.write("Prediction :  ", list_col[1]['Prediction'])
                with colo3:
                    st.write("Loan ID :  ", list_col[2]['Loan_ID'])
                    st.write("Prediction :  ", list_col[2]['Prediction'])
                with colo4:
                    st.write("Loan ID :  ", list_col[3]['Loan_ID'])
                    st.write("Prediction :  ", list_col[3]['Prediction'])
                st.divider()
                list_col = []