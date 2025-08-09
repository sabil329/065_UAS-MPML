# app.py - Versi Streamlit
import streamlit as st
import joblib
import pandas as pd

# =======================
# Load model & scaler
# =======================
model = joblib.load("output_models/best_model.joblib")
scaler = joblib.load("output_models/scaler.joblib")
top20_features = joblib.load("output_models/top20_features.joblib")

# =======================
# Judul aplikasi
# =======================
st.set_page_config(page_title="Prediksi Kelulusan Siswa", page_icon="🎓", layout="centered")
st.title("🎓 Prediksi Kelulusan Siswa")

# =======================
# Form input
# =======================
with st.form("prediksi_form"):
    nama = st.text_input("Nama", placeholder="Masukkan nama lengkap")
    kelas = st.text_input("Kelas", placeholder="Contoh: XI IPA 1")
    absences = st.number_input("Nomor Absen (Jumlah ketidakhadiran)", min_value=0, max_value=100, step=1)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
    submitted = st.form_submit_button("🔍 Prediksi")

# =======================
# Proses prediksi
# =======================
if submitted:
    # Buat dictionary default semua fitur = 0
    default_values = {col: 0 for col in top20_features}
    default_values["absences"] = absences
    default_values["sex_M"] = 1 if jenis_kelamin == "Laki-laki" else 0

    # Urutkan sesuai fitur model
    df_input = pd.DataFrame([[default_values[col] for col in top20_features]], columns=top20_features)
    X_scaled = scaler.transform(df_input)

    # Prediksi
    prediction = model.predict(X_scaled)[0]
    prob = round(model.predict_proba(X_scaled)[0][1] * 100, 2)
    result = "Lulus ✅" if prediction == 1 else "Tidak Lulus ❌"

    # =======================
    # Tampilkan hasil
    # =======================
    st.subheader("📊 Hasil Prediksi")
    st.markdown(f"**Nama:** {nama}")
    st.markdown(f"**Kelas:** {kelas}")
    st.markdown(f"**Prediksi:** {result}")
    st.markdown(f"**Probabilitas Lulus:** {prob}%")

    # Notifikasi visual
    if prediction == 1:
        st.success(f"Selamat {nama}, kamu diprediksi akan **Lulus** 🎉")
    else:
        st.error(f"Maaf {nama}, kamu diprediksi **Tidak Lulus** 😔")

# =======================
# Catatan
# =======================
st.markdown("---")
st.caption("Aplikasi ini dibuat untuk UAS MPML. Model dilatih menggunakan dataset student_data.csv dan menampilkan prediksi kelulusan berdasarkan variabel: Nama, Kelas, Nomor Absen, dan Jenis Kelamin.")
