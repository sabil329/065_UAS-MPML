from flask import Flask, request, render_template_string
import joblib
import pandas as pd

# Load model dan scaler
model = joblib.load("output_models/best_model.joblib")
scaler = joblib.load("output_models/scaler.joblib")
top20_features = joblib.load("output_models/top20_features.joblib")

app = Flask(__name__)

# HTML form sederhana
html_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediksi Kelulusan Siswa</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f8f9fa; }
        .container {
            max-width: 400px; margin: 40px auto; background: white; padding: 20px;
            border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h2 { text-align: center; }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input, select {
            width: 100%; padding: 8px; margin-top: 5px;
            border: 1px solid #ccc; border-radius: 5px;
        }
        button {
            margin-top: 15px; width: 100%; padding: 10px;
            background: #007bff; color: white; border: none; border-radius: 5px;
            font-size: 16px;
        }
        .result { margin-top: 15px; text-align: center; padding: 10px; border-radius: 5px; }
        .pass { background-color: #d4edda; color: #155724; }
        .fail { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Prediksi Kelulusan Siswa</h2>
        <form method="post" action="/predict">
            <label>Nama:</label>
            <input type="text" name="nama" required>

            <label>Kelas:</label>
            <input type="text" name="kelas" required>

            <label>Nomor Absen:</label>
            <input type="number" name="absences" required>

            <label>Jenis Kelamin:</label>
            <select name="sex_M">
                <option value="1">Laki-laki</option>
                <option value="0">Perempuan</option>
            </select>

            <button type="submit">Prediksi</button>
        </form>

        {% if result is defined %}
            <div class="result {{ 'pass' if result == 'Lulus' else 'Tidak Lulus' }}">
                <strong>{{ nama }} ({{ kelas }})</strong><br>
                Hasil: {{ result }}<br>
                Probabilitas Lulus: {{ prob }}%
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(html_form)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        nama = request.form.get('nama')
        kelas = request.form.get('kelas')
        absences = float(request.form.get('absences', 0))
        sex_M = float(request.form.get('sex_M', 0))

        # Default semua fitur ke 0, lalu isi yang diinput
        default_values = {col: 0 for col in top20_features}
        if 'absences' in default_values: default_values['absences'] = absences
        if 'sex_M' in default_values: default_values['sex_M'] = sex_M 

        # Data sesuai urutan fitur
        df_input = pd.DataFrame([[default_values[col] for col in top20_features]], columns=top20_features)

        # Scaling
        X_scaled = scaler.transform(df_input)

        # Prediksi
        prediction = model.predict(X_scaled)[0]
        prob = round(model.predict_proba(X_scaled)[0][1] * 100, 2)
        result = "Lulus" if prediction == 1 else "Tidak Lulus"

        return render_template_string(html_form, result=result, prob=prob, nama=nama, kelas=kelas)
    except Exception as e:
        return f"<h3>Error: {e}</h3>"

if __name__ == "__main__":
    app.run(debug=True)
