import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = pickle.load(open('model.pkl','rb'))

def predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, BMI, DPF, age):
    input = np.array([[pregnancies, glucose, bloodpressure, skinthickness, insulin, BMI, DPF, age]]).astype(np.float64)
    prediction = model.predict_proba(input)
    return prediction

def main():
    st.title('Diabetes Prediction')
    html_temp = """
    <style>
    body {
    color: #fff;
    background-color: #9adeed;
    }
    <\style>
    """
    # st.balloons()

    st.markdown(html_temp, unsafe_allow_html=True)

    glucose = st.text_input("Glucose Level")
    bloodpressure = st.text_input("Blood Pressure")
    skinthickness = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DPF = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")
    pregnancies = st.text_input("No. of Pregnancies")

    safe_html = """
    <div style = "background-color:#F4D03F; padding:10px">
    <h2 style = "color:white; text-align:center;"> You are safe.</h2>
    <\div>
    """

    danger_html = """
    <div style = "background-color:#F08080; padding:10px">
    <h2 style = "color:white; text-align:center;"> You are NOT safe.</h2>
    <\div>
    """
    if st.button("Predict"):
        output = predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, BMI, DPF, age)
        text = 'The probability that you have diabetes is : '+str(output[0][1]*100)+'%'
        st.success(text)
        #print(output)
        # if output[0][1] > 0.5:
        #     st.markdown(safe_html,unsafe_allow_html=True)
        # else:
        #     st.markdown(danger_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
