# app.py - Versi Streamlit dengan auto-train kalau file model tidak ada
import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =======================
# Cek & load atau buat model
# =======================
if not (os.path.exists("best_model.joblib") and os.path.exists("scaler.joblib") and os.path.exists("top20_features.joblib")):
    st.warning("⚠️ File model tidak ditemukan. Melatih ulang model...")

    # Load dataset
    df = pd.read_csv("student_data.csv")
    df["Pass"] = (df["G3"] >= 10).astype(int)

    # Ambil fitur yang digunakan
    X = df[["absences", "sex"]].copy()
    X["sex_M"] = (X["sex"] == "M").astype(int)  # Encode gender M=1
    X = X.drop(columns=["sex"])
    top20_features = X.columns.tolist()
    y = df["Pass"]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Train model sederhana
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Simpan model
    joblib.dump(model, "best_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(top20_features, "top20_features.joblib")

else:
    # Kalau file sudah ada, langsung load
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    top20_features = joblib.load("top20_features.joblib")

# =======================
# Judul aplikasi
# =======================
st.set_page_config(page_title="Prediksi Kelulusan Siswa", page_icon="🎓", layout="centered")
st.title("🎓 Prediksi Kelulusan Siswa")
st.markdown("Masukkan data siswa untuk memprediksi kelulusan berdasarkan absensi dan jenis kelamin.\n")

# =======================
# Form input
# =======================
with st.form("prediksi_form"):
    nama = st.text_input("Nama", placeholder="Masukkan nama lengkap")
    kelas = st.text_input("Kelas", placeholder="Contoh: 11 IPA 1")
    absences = st.number_input("Nomor Absen", min_value=0, max_value=100, step=1)
    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

    submitted = st.form_submit_button("Prediksi")

# =======================
# Proses prediksi dan tampilkan hasil
# =======================
if submitted:
    if nama.strip() == "":
        st.error("❌ Mohon isi nama lengkap siswa.")
    elif kelas.strip() == "":
        st.error("❌ Mohon isi kelas siswa.")
    else:
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

        # Tampilkan hasil dengan pemformatan menarik
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        st.markdown(f"**Nama :** `{nama}`")
        st.markdown(f"**Kelas :** `{kelas}`")
        st.markdown(f"**Jenis Kelamin :** `{jenis_kelamin}`")
        st.markdown(f"**Prediksi Kelulusan :** **{result}**")
        st.markdown(f"**Probabilitas Lulus :** **{prob} %**\n")

        if prediction == 1:
            st.success(f"🎉 Selamat {nama}, kamu diprediksi **Lulus**. Terus pertahankan semangat belajar!")
            st.balloons()
        else:
            st.error(f"😔 Maaf {nama}, kamu diprediksi **Tidak Lulus**. Jangan putus asa, coba tingkatkan usaha dan konsultasi dengan guru.")

# =======================
# Catatan
# =======================
st.markdown("---")
st.caption("Aplikasi ini dibuat untuk UAS MPML oleh Salsabilla Nikita Untoro.")
