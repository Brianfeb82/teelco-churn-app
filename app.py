import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- LOAD MODEL & SCALER ---
# Ini tahap mengambil 'Tupperware' yang sudah kamu buat tadi
model = pickle.load(open('model_churn.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="AI Churn Predictor", page_icon="📊")

st.title("📊 Telco Churn Intelligence")
st.markdown("Aplikasi ini menggunakan model **Random Forest** untuk memprediksi apakah pelanggan akan berhenti berlangganan.")

# --- INPUT USER ---
st.sidebar.header("Data Pelanggan")

# Kita buat input sederhana untuk demo (sesuaikan dengan fitur utama kamu)
tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)

# Karena model Random Forest kamu sebelumnya mungkin punya 30+ kolom (akibat get_dummies),
# Kita harus membuat data input yang punya jumlah kolom yang SAMA.
# Triknya: Buat array berisi nol sebanyak jumlah fitur aslimu.

if st.button("Analisis Status Pelanggan"):
    # Misalkan fitur asli kamu ada 30 kolom (cek dengan X.shape[1] di notebook)
    # Untuk contoh ini, kita buat dummy input agar tidak error:
    input_data = np.zeros((1, model.n_features_in_)) 

    # Isi kolom pertama dan kedua (tenure & monthly charges) 
    # Sesuaikan urutannya dengan urutan kolom di X_train kamu
    input_data[0, 0] = tenure
    input_data[0, 1] = monthly_charges
    input_data[0, 2] = total_charges

    # Scaling & Prediksi
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()

    if prediction[0] == 1:
        st.error(f"### ⚠️ HASIL: CHURN (RISIKO TINGGI)")
        st.write(f"Probabilitas kabur: **{probability*100:.2f}%**")
        st.write("👉 **Saran:** Segera tawarkan kontrak tahunan atau diskon paket.")
    else:
        st.success(f"### ✅ HASIL: SETIA (LOYAL)")
        st.write(f"Probabilitas tetap: **{(1-probability)*100:.2f}%**")
        st.write("👉 **Saran:** Jaga layanan agar tetap stabil.")
