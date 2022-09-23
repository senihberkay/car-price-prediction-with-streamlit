
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

st.markdown("<h2 style='text-align:center; color:Black;'>Car Price Prediction</h2>", unsafe_allow_html=True)

df = pd.read_csv("final_scout.csv")

features = ["make_model", "hp_kW", "km","age", "Gearing_Type", "Gears","price"]
df = df[features]

if st.checkbox("Show Dataframe"):
    st.write(df.head())

X = df.drop(columns = ["price"])
y = df.price

cat = X.select_dtypes("object").columns
cat = list(cat)

column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), 
                                       remainder=MinMaxScaler())

operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso(alpha = 0.01))]
pipe_model = Pipeline(steps=operations)
pipe_model.fit(X, y)

pickle.dump(pipe_model, open('autoscout_project.pkl', 'wb'))
 
st.sidebar.title("Please select the features you want for price estimation")

features = ["make_model", "hp_kW", "km","age", "Gearing_Type", "Gears","price"]

def user_input_features() :
    make_model = st.sidebar.selectbox("Make Model", ("Audi A3","Audi A1","Opel Insignia", "Opel Astra", "Opel Corsa", "Renault Clio", "Renault Espace", "Renault Duster"))
    Gearing_Type = st.sidebar.selectbox("Gearing Type", ("Manual","Automatic", "Semi-automatic"))
    age = st.sidebar.number_input("Age:",min_value=0, max_value=3)
    Gears = st.sidebar.radio("Gears",(5,6,7,8))
    hp_kW = st.sidebar.slider("Horse Power(kW)", df["hp_kW"].min(), df["hp_kW"].max(), float(df["hp_kW"].median()),1.0)
    km = st.sidebar.slider("Kilometer(km)", df["km"].min(), df["km"].max(), float(df["km"].median()),1.0)
    data = {"make_model" : make_model,
            "Gearing_Type" : Gearing_Type,
            "age" : age,
            "hp_kW" : hp_kW,
            "km" : km,
            "Gears" : Gears}
    features = pd.DataFrame(data, index=[0])
    return features



input_df = user_input_features()
st.success("Selected Features")
st.write(input_df)

model = pickle.load(open("final_model_scout.pkl", "rb"))

#Apply model to make predictions
if st.button('Make Prediction'):
    st.success(f'Analyze Predict:&emsp;{model.predict(input_df)[0].round(2)}')
