import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

# Configuration de la page
st.set_page_config(page_title="Modélisation Excel", layout="wide")

# Titre
st.title("📈 Application de Modélisation - Fichier Excel avec Graphes, Filtres et Choix de Méthode")

# Upload du fichier
uploaded_file = st.file_uploader("📤 Chargez votre fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Lecture Excel
        data = pd.read_excel(uploaded_file)
        st.subheader("📄 Aperçu des données")
        st.dataframe(data, use_container_width=True)

        st.sidebar.header("🔎 Paramètres de Modélisation")

        # Choix du modèle
        modele_selectionne = st.sidebar.selectbox(
            "Choisissez votre méthode de modélisation :",
            ("Régression Linéaire", "Random Forest Regressor")
        )

        # Colonnes disponibles
        colonnes = data.columns.tolist()

        # Sélection de colonnes
        x_cols = st.sidebar.multiselect("Sélectionnez les colonnes X (Variables indépendantes)", colonnes)
        y_col = st.sidebar.selectbox("Sélectionnez la colonne Y (Variable cible)", colonnes)

        if x_cols and y_col:
            X = data[x_cols]
            y = data[y_col]

            # Vérification taille des données
            if len(data) < 2:
                st.warning("⚠️ Pas assez de données pour split train/test. Modèle sur toutes les données.")
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Création du modèle selon le choix
            if modele_selectionne == "Régression Linéaire":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Entrainement du modèle
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

            # Affichage des métriques
            st.subheader(f"📊 Résultats du modèle ({modele_selectionne})")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2:.3f}" if not np.isnan(r2) else "Non défini")
            with col2:
                st.metric("MAE", f"{mae:.3f}")
            with col3:
                st.metric("RMSE", f"{rmse:.3f}")

            # Comparaison réelle vs prédite
            st.subheader("🔍 Comparaison Réelle vs Prédite")
            resultat = pd.DataFrame({
                "Réel": y_test,
                "Prédit": y_pred
            })
            st.dataframe(resultat, use_container_width=True)

            # Graphes
            st.subheader("📈 Graphiques")

            # Scatter Plot (Réel vs Prédit)
            fig1, ax1 = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
            ax1.set_xlabel("Valeurs Réelles")
            ax1.set_ylabel("Valeurs Prédites")
            ax1.set_title(f"Réel vs Prédit ({modele_selectionne})")
            st.pyplot(fig1)

            # Heatmap des corrélations
            st.subheader("🧠 Heatmap des Corrélations")
            fig2, ax2 = plt.subplots()
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
            st.pyplot(fig2)

        else:
            st.info("ℹ️ Sélectionnez au moins une colonne X et une colonne Y dans la barre latérale.")

    except Exception as e:
        st.error(f"🚨 Erreur lors de la lecture du fichier : {e}")
else:
    st.info("📥 Veuillez charger un fichier Excel pour démarrer.")
