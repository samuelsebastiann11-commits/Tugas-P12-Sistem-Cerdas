import streamlit as st
import pandas as pd
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Perkiraan Biaya Kos", page_icon="🏡", layout="wide")
st.title("Aplikasi Perkiraan Biaya Kos")
st.write("Dibuat oleh: Samuel Aldrik Sebastian (253307057)")
st.write("Aplikasi untuk membantu membandingkan perkiraan biaya kos di Madiun.")

# Load Model
@st.cache_resource
def muat_model():
    return joblib.load("model_kos.joblib")

# Load Dataset
@st.cache_data
def muat_data():
    url = "https://raw.githubusercontent.com/samuelsebastiann11-commits/Sistem-Cerdas/refs/heads/master/kos_madiun.csv"
    data = pd.read_csv(url)
    data.columns = data.columns.str.strip().str.lower()
    for kolom in ["jarak_km", "rating", "harga"]:
        data[kolom] = pd.to_numeric(data[kolom], errors="coerce")
    data["ac"] = data["ac"].astype(str).str.strip().str.lower()
    return data.dropna(subset=["jarak_km", "rating", "harga", "ac"])

model = muat_model()
data = muat_data()

st.subheader("Data Kos Madiun")
st.dataframe(data, use_container_width=True)

# Pilihan referensi data
pilihan = st.selectbox(
    "Pilih data kos sebagai referensi", 
    data.index.tolist(),
    format_func=lambda i: str(data.loc[i, "nama"]) if "nama" in data.columns else f"Data kos {i+1}"
)

baris = data.loc[pilihan]
col1, col2 = st.columns(2)
with col1:
    st.metric("Jarak ke kampus", f"{baris['jarak_km']:.1f} km")
    st.metric("Rating", f"{baris['rating']:.1f} / 5")
with col2:
    st.metric("Fasilitas AC", str(baris["ac"]).upper())
    st.metric("Harga pada dataset", f"Rp {baris['harga'] * 1000:,.0f}")

st.subheader("Coba Prediksi dengan Kondisi Pilihan")
jarak = st.number_input("Jarak ke kampus (km)", min_value=0.0, value=float(baris["jarak_km"]), step=0.1)
rating = st.number_input("Rating kos", min_value=0.0, max_value=5.0, value=float(baris["rating"]), step=0.1)
ac = st.selectbox("Fasilitas AC", ["ya", "tidak"], index=0 if baris["ac"] == "ya" else 1)

if st.button("Hitung Perkiraan Biaya"):
    data_baru = pd.DataFrame({
        "jarak_km": [jarak],
        "rating": [rating],
        "ac_num": [1 if ac == "ya" else 0]
    })
    hasil = float(model.predict(data_baru)[0])
    st.success(f"Perkiraan biaya: Rp {hasil * 1000:,.0f} per bulan")

st.subheader("Rata-rata Harga Berdasarkan Fasilitas AC")
rata_ac = data.groupby("ac")["harga"].mean().reindex(["ya", "tidak"]).dropna()
rata_ac.index = ["Dengan AC" if x == "ya" else "Tanpa AC" for x in rata_ac.index]
st.bar_chart(rata_ac * 1000)
st.caption("Grafik menampilkan rata-rata harga berdasarkan data pada dataset.")