import streamlit as st
import pandas as pd
import joblib

st.header("Streamlit Classification App")

height = st.number_input("Enter Height")

weight = st.number_input("Enter Weight")

eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

if st.button("Submit"):

    clf = joblib.load("clf.pkl")


    X = pd.DataFrame([[height, weight, eyes]],
                     columns=["Height", "Weight", "Eyes"])
    X = X.replace(["Brown", "Blue"], [1, 0])

    prediction = clf.predict(X)[0]

    st.text(f"This instance is a {prediction}")