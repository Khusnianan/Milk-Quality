import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')  # scaler untuk 7 fitur
label_encoder = joblib.load('label_encoder.pkl')  # encoder untuk grade: high, medium, low

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="milk_grade_predict.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Milk Quality Grade Prediction")
st.write("Masukkan data kualitas susu untuk memprediksi grade susu (High, Medium, Low).")

# Input pengguna
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=6.6)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=35.0)
taste = st.selectbox("Taste", [0, 1])  # 0 = Bad, 1 = Good
odor = st.selectbox("Odor", [0, 1])    # 0 = Bad, 1 = Good
fat = st.selectbox("Fat", [0, 1])      # 0 = Low, 1 = High
turbidity = st.selectbox("Turbidity", [0, 1])  # 0 = Low, 1 = High
colour = st.slider("Colour", 240, 255, 254)

# Tombol prediksi
if st.button("Prediksi Grade Susu"):
    # Preprocessing
    input_data = np.array([[ph, temperature, taste, odor, fat, turbidity, colour]])
    input_scaled = scaler.transform(input_data).astype(np.float32)

    # Prediksi dengan TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_label = np.argmax(prediction)

    # Cek jika label yang diprediksi valid
    if predicted_label >= len(label_encoder.classes_):
        st.error("Model memprediksi label yang tidak dikenal. Mungkin model tidak sinkron dengan encoder.")
    else:
        grade = label_encoder.inverse_transform([predicted_label])[0]
        st.success(f"Grade susu yang diprediksi: **{grade.upper()}**")
