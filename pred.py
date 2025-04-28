import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

# Titre
st.title("📈 Application de Modélisation - Fichier Excel")

# Chargement du fichier Excel
uploaded_file = st.file_uploader("📤 Chargez votre fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        data = pd.read_excel(uploaded_file)
        st.subheader("📄 Aperçu des données")
        st.write(data.head())

        st.sidebar.subheader("🔧 Paramètres")

        # Sélection des colonnes
        colonnes = data.columns.tolist()
        x_cols = st.sidebar.multiselect("Sélectionnez les colonnes X", colonnes)
        y_col = st.sidebar.selectbox("Sélectionnez la colonne Y (à prédire)", colonnes)

        if x_cols and y_col:
            X = data[x_cols]
            y = data[y_col]

            if len(X) == 0:
                st.error("🚫 Aucune donnée disponible après sélection. Vérifiez vos colonnes.")
            else:
                if len(X) < 2:
                    st.warning("⚠️ Pas assez de données pour faire un split train/test. Utilisation de toutes les données pour l'entraînement.")
                    X_train, X_test, y_train, y_test = X, X, y, y
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Création et entraînement du modèle
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Prédiction
                y_pred = model.predict(X_test)

                # Calcul des métriques
                if len(y_test) >= 2:
                    r2 = r2_score(y_test, y_pred)
                else:
                    r2 = float('nan')

                mae = mean_absolute_error(y_test, y_pred)
                rmse = sqrt(mean_squared_error(y_test, y_pred))

                # Résultats
                st.subheader("📊 Résultats du Modèle")
                if np.isnan(r2):
                    st.write("**R² Score** : Non défini (moins de 2 échantillons)")
                else:
                    st.write(f"**R² Score** : {r2:.3f}")

                st.write(f"**MAE** (Erreur Absolue Moyenne) : {mae:.3f}")
                st.write(f"**RMSE** (Erreur Quadratique Moyenne) : {rmse:.3f}")

                # Comparaison réelle vs prédite
                st.subheader("🔍 Comparaison Réelle vs Prédit")
                resultats = pd.DataFrame({"Valeur Réelle": y_test, "Valeur Prédit": y_pred})
                st.write(resultats)
        else:
            st.info("ℹ️ Veuillez sélectionner au moins une colonne pour X et une pour Y.")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")

else:
    st.info("📥 Veuillez charger un fichier Excel pour commencer.")

