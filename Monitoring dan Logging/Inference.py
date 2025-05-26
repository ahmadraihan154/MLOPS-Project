import streamlit as st
import requests

st.title("Prediksi Harga Berlian")

carat = st.number_input(label='carat (weight diamond)', min_value=0, max_value=5, value=None, placeholder='Masukkan angka dari 0 s.d 5')
x = st.number_input("X (lenght diamond)", 0.0, 10.0, value=None, placeholder='Masukkan angka antara 0 s.d 10')
y = st.number_input("Y (width diamond)", 0.0, 10.0, value=None, placeholder='Masukkan angka antara 0 s.d 10')
z = st.number_input("Z (depth diamond)", 0.0, 10.0, value=None, placeholder='Masukkan angka antara 0 s.d 10')
table = st.number_input("table (diamond proportion)", 50.0, 75.0, value=None, placeholder='Masukkan angka antara 50 s.d 75')
cut = st.selectbox("Cut (dari fair (terburuk) s.d Ideal (terbaik))", ["Ideal", "Premium", "Good", "Very Good", "Fair"])
color = st.selectbox("Color (dari J (terburuk) s.d D (terbaik))", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity (dari I1 (terburuk) s.d IF (terbaik))", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])

if st.button("Prediksi"):
    # Validasi semua field terisi
    if None in [carat, x, y, z, table] or None in [cut, color, clarity]:
        st.error("Harap isi semua field terlebih dahulu!")
    else:
        input_data = {
            "carat": float(carat),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "table": float(table),
            "cut": cut,
            "color": color,
            "clarity": clarity
        }
        
        try:
            with st.spinner('Sedang memproses prediksi...'):
                response = requests.post("http://localhost:8000/predict", 
                                       json=input_data, 
                                       timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Harga Prediksi:** ${result['predicted_price']:,.2f}")
                
                # Tampilkan detail input
                with st.expander("Detail Input"):
                    st.json(input_data)
                
            else:
                st.error(f"Error dari server: {response.text}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"Koneksi ke server gagal: {str(e)}")