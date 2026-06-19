import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Model Imports
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Fake Deployment", page_icon="🎯", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("ML")
task = st.sidebar.radio("ML:", ["Regresi", "Klasifikasi"])
st.sidebar.markdown("---")
st.title(f"{task}")




# =====================================================================
# 📈 PILAR REGRESI
# =====================================================================
if task == "Regresi":
    model_choice = st.selectbox("Pilih Model & Dataset Regresi:", ["Linear Regression"])
    st.markdown("---")

    # --- MODEL 1: LINEAR REGRESSION ---
    if model_choice == "Linear Regression":
        st.subheader("Linear Regression Engine")
        csv_file = "regression_data.csv"  # Pushed to your Git repo

        if not os.path.exists(csv_file):
            st.error(f"❌ File `{csv_file}` tidak ditemukan di repositori Git Anda.")
        else:
            df = pd.read_csv(csv_file)

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer.fit(X[:, 1:3])
            X[:, 1:3] = imputer.transform(X[:, 1:3])



            # Splitting
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # 2. TRAINING MODEL
            model = LinearRegression()
            model.fit(X_train, y_train)
            with st.expander("Lihat Dataset"):
                st.dataframe(df.head(5))

            # Form Input untuk User
            st.markdown("### 🎛️ Form Prediksi Baru")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                in_AT = st.number_input("Ambient Temperature", value=5)
            with col2:
                in_V = st.number_input("Exhaust Vacum", value=50)
            with col3:
                in_AP = st.number_input("Ambient Pressure", value=1010)
            with col4:
                in_RH = st.number_input("Relative Humidity", value=50.0)

            # Trigger Pipeline ONLY when button is pressed
            if st.button("🚀 Prediction", type="primary"):
                pred = model.predict([[in_AT, in_V, in_AP, in_RH]])
                st.write("Prediction: ", pred, st.write("R2 Score :", r2_score(y_train, model.predict(X_train))))

# =====================================================================
# 🗂️ PILAR KLASIFIKASI
# =====================================================================
elif task == "Klasifikasi":
    model_choice = st.selectbox("Pilih Model & Dataset Klasifikasi:",
                                ["Data Medis"])
    st.markdown("---")

    # --- MODEL 1: LOGISTIC REGRESSION (NEEDS SCALING) ---
    if model_choice == "Data Medis":
        st.subheader("Klasifikasi Dataset Kanker")
        csv_file = "classification_data.csv"

        if not os.path.exists(csv_file):
            st.error(f"❌ File `{csv_file}` tidak ditemukan di repositori Git Anda.")
        else:
            df = pd.read_csv(csv_file)

            with st.expander("Lihat Dataset"):
                st.dataframe(df.head(5))

            st.markdown("### 🎛️ Form Data Klinis Pasien")
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
            with c1:
                in_thick = st.number_input("Clump Thickness", value=3)
            with c2:
                in_size = st.number_input("Uniformity of Cell Size", value=4)
            with c3:
                in_shape = st.number_input("Uniformity of Cell Shape", value=2)
            with c4:
                in_adhesion = st.number_input("Marginal Adhesion", value=3)
            with c5:
                in_epithelial = st.number_input("Single Epithelian Cell Size", value=2)
            with c6:
                in_nuclei = st.number_input("Bare Nuclei", value=4)
            with c7:
                in_chromatin = st.number_input("Bland Chromatin", value=5)
            with c8:
                in_nucleoli = st.number_input("Normal Nucleoli", value=2)
            with c9:
                in_mitoses = st.number_input("Mitoses", value=2)

            if st.button("🚀 Analisis Risiko Kesehatan", type="primary"):
                # 1. PIPELINE PREPROCESSING KHUSUS (Imputasi Mean + Scaling)
                X = df.iloc[:, 1:10]
                y = df.iloc[:, -1]

                # 2. TRAINING
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                model = XGBClassifier()
                y_train[y_train == 2] = 0
                y_train[y_train == 4] = 1
                model.fit(X_train, y_train)

                # 3. PREDIKSI
                accuracies = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)
                st.write("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
                st.write("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
                pred = model.predict([[in_thick, in_size, in_shape, in_adhesion, in_epithelial, in_nuclei, in_chromatin, in_nucleoli, in_mitoses]])
                if pred == 0:
                    pred = "Benign"
                else:
                    pred = "Malignant"
                st.warning(f"Hasil Diagnosis Model AI: Pasien Terkategori **{pred}**")
