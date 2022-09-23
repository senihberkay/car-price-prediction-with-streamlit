import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

html_temp = """
<div style="background-color:tomato;padding:1.5px">
<h2 style="color:white;text-align:center;">Used Car Price Prediction</h2>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)


html_temp = """
<div style="background-color:yellow;padding:1.5px">
<h2 style="color:black;text-align:center;"> Arguments for the Prediction </h2>
</div><br>"""
st.sidebar.markdown(html_temp,unsafe_allow_html=True)


st.success("Please use the sidebar at left-hand-side to input the parameters of the price predictor")
st.markdown("**And select the model of your car (from the left sidebar)**")
st.markdown("**1-** Audi A3, **2-** Audi A1, **3-** Opel Insignia, **4-** Opel Astra, **5-** Opel Corsa")
st.markdown("**6-** Renault Clio, **7-** Renault Espace, **8-** Renault Duster, **9-** Audi A2")
make_model = st.sidebar.radio("Make & Model of Your Car:",(1,2,3,4,5,6,7,8,9))


filename = 'xgboost_model.pkl'
model = pickle.load(open(filename, 'rb'))

gears = st.sidebar.slider("The 'gear type' of your car:",min_value=1, max_value=7, value=5, step=1)
age = st.sidebar.slider("The 'age' of your car:",min_value=1, max_value=50, value=3, step=1)
hp = st.sidebar.slider("'Horse Power' of your car::",min_value=1, max_value=294, value=100, step=1)
km = st.sidebar.slider("Kilometers travelled:",min_value=0, max_value=317000, value=10000, step=10)
dict = {}
dict["gears"] = gears
dict["hp"] = hp
dict["age"] = age
dict["make_model"] = make_model
dict["km"] = km
df = pd.DataFrame.from_dict([dict])
st.write(df)


if st.button("Predict"):
    pred = model.predict(df)
    st.write(pred[0])

