import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("heart_model.pkl")


st.markdown("<h1 style='text-align: center;'>🫀 Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.write("Enter Information About U To Predict Heart Disease")


st.markdown("""
<style>
@keyframes pulse {
0% { transform: scale(1); }
50% { transform: scale(1.3); }
100% { transform: scale(1); }
}
.heartbeat {
width: 60px;
height: 60px;
background-color: red;
border-radius: 50%;
margin: auto;
animation: pulse 1s infinite;
}
.prob-bar {
height: 30px;
border-radius: 15px;
}
</style>
<div class="heartbeat"></div>
""", unsafe_allow_html=True)


age = st.slider("العمر (Age)", 1, 120, 52)
sex = st.selectbox("الجنس (Sex)", [0, 1], index=1)
cp = st.selectbox("نوع ألم الصدر (Chest Pain Type - cp)", [0, 1])
trestbps = st.slider("ضغط الدم أثناء الراحة (Resting Blood Pressure)", 80, 200, 125)
chol = st.slider("الكوليسترول (Cholesterol)", 100, 400, 212)
fbs = st.selectbox("سكر الدم الصائم > 120 (Fasting Blood Sugar > 120)", [0, 1])
restecg = st.selectbox("تخطيط القلب أثناء الراحة (Resting ECG)", [0, 1])
thalach = st.slider("أقصى معدل لضربات القلب (Max Heart Rate)", 70, 220, 168)
exang = st.selectbox("ذبحة صدرية ناتجة عن التمرين (Exercise Induced Angina)", [0, 1])
oldpeak = st.slider("انخفاض مقطع ST (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
slope = st.selectbox("ميل مقطع ST (Slope of ST)", [0, 1])
ca = st.selectbox("عدد الأوعية الدموية الرئيسية (Number of Major Vessels - Ca)", [0, 1, 2, 3])
thal = st.selectbox("اختبار الثلاسيميا (Thal)", [0, 1])


input_df = pd.DataFrame([[
    age, sex, cp, trestbps, chol, fbs,
    restecg, thalach, exang, oldpeak,
    slope, ca, thal
]], columns=['Age','Sex','Cp','Trestbps','Chol','Fbs',
            'Restecg','Thalach','Exang','Oldpeak',
            'Slope','Ca','Thal'])


if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:,1][0]  
    bar_color = 'red' if probability > 0.5 else 'green'
    st.markdown(f"""
    <div style="background-color: lightgray; border-radius: 15px;">
    <div class="prob-bar" style="width: {probability*100:.1f}%; background-color: {bar_color}; text-align:center; color:white;">
        {probability*100:.2f}%
    </div>
    </div>
    """, unsafe_allow_html=True)

    if prediction[0] == 1:
        st.error("⚠️ Patient has possibility of high Heart presentation")
    else:

        st.success("✅ The patient is healthy, the probability of the disease is low ")
