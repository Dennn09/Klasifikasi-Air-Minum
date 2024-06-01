import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Judul aplikasi
st.title('Analisis Kelayakan Air')

# Teks markdown
st.markdown("""
Aplikasi ini memprediksi kelayakan air berdasarkan beberapa sifat kimia.
""")

# Memuat data
@st.cache
def load_data():
    df = pd.read_csv('water_potability.csv')
    return df

df = load_data()

# Praproses data
imputer = SimpleImputer(strategy='mean')
X = df.drop(['Potability'], axis=1)
y = df['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_filled = imputer.fit_transform(X_train)
X_test_filled = imputer.transform(X_test)

# Pelatihan model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_filled, y_train)

# Form prediksi kelayakan air
st.header('Prediksi Kelayakan Sampel Air Anda')
ph = st.number_input('Masukkan tingkat pH')
hardness = st.number_input('Masukkan tingkat Kekerasan')
solids = st.number_input('Masukkan tingkat Padatan')
chloramines = st.number_input('Masukkan tingkat Kloramin')
sulfate = st.number_input('Masukkan tingkat Sulfat')
conductivity = st.number_input('Masukkan tingkat Konduktivitas')
organic_carbon = st.number_input('Masukkan tingkat Karbon Organik')
trihalomethanes = st.number_input('Masukkan tingkat Trihalometana')
turbidity = st.number_input('Masukkan tingkat Kekeruhan')

if st.button('Prediksi Kelayakan'):
    test_data = pd.DataFrame({
        'ph': [ph], 'Hardness': [hardness], 'Solids': [solids],
        'Chloramines': [chloramines], 'Sulfate': [sulfate], 'Conductivity': [conductivity],
        'Organic_carbon': [organic_carbon], 'Trihalomethanes': [trihalomethanes], 'Turbidity': [turbidity]
    })
    test_data_imputed = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
    pred = model.predict(test_data_imputed)
    st.write('Kelayakan air yang diprediksi adalah: ' + ('Layak' if pred[0] == 1 else 'Tidak Layak'))
