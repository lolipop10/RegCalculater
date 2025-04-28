import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration de la page
st.set_page_config(page_title="📈 Modélisation de Régression", layout="wide")

# Titre principal
st.title("📈 Application de Modélisation de Régression")

# Upload du fichier
uploaded_file = st.file_uploader("📤 Charger un fichier Excel", type=["xlsx"])

if uploaded_file:
    # Lecture du fichier
    df = pd.read_excel(uploaded_file)
    
    st.subheader("🗂️ Aperçu des données chargées :")
    st.dataframe(df.head())

    # Zone de filtre sur la sidebar
    st.sidebar.header("🧰 Filtrage des Données")
    filtre_colonne = st.sidebar.selectbox("Choisir une colonne pour filtrer :", options=["-- Aucun filtre --"] + df.columns.tolist())

    if filtre_colonne != "-- Aucun filtre --":
        unique_values = df[filtre_colonne].dropna().unique().tolist()
        selected_value = st.sidebar.selectbox(f"Valeur de '{filtre_colonne}' :", unique_values)
        df = df[df[filtre_colonne] == selected_value]

    # Sélection des variables
    st.sidebar.header("🔧 Sélection des Variables")
    colonnes = df.columns.tolist()
    x_cols = st.sidebar.multiselect("Variables explicatives (X)", colonnes)
    y_col = st.sidebar.selectbox("Variable à prédire (Y)", colonnes)

    # Sélection du modèle
    st.sidebar.header("⚙️ Modèle de Régression")
    model_choice = st.sidebar.selectbox("Choisir le modèle :", ["Régression Linéaire", "Random Forest"])

    if x_cols and y_col:
        # Préparation des données
        data = df[x_cols + [y_col]].dropna()
        X = data[x_cols]
        y = data[y_col]

        # Séparation en train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Modèle
        if model_choice == "Régression Linéaire":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5  # Pas d'argument squared pour éviter les erreurs de compatibilité

        st.subheader("📊 Résultats du Modèle")
        st.metric("🎯 R² (Coefficient de Détermination)", f"{r2:.3f}")
        st.metric("📏 MAE (Erreur Absolue Moyenne)", f"{mae:.3f}")
        st.metric("📐 RMSE (Erreur Quadratique Moyenne)", f"{rmse:.3f}")

        # Affichage dynamique
        st.subheader("📉 Graphique Réel vs Prédit")

        results_df = pd.DataFrame({
            "Valeur Réelle": y_test,
            "Valeur Prédite": y_pred
        })

        fig = px.scatter(
            results_df,
            x="Valeur Réelle",
            y="Valeur Prédite",
            title="Valeur Réelle vs Valeur Prédite",
            labels={"Valeur Réelle": "Y Réel", "Valeur Prédite": "Y Prédit"},
            trendline="ols",
            color_discrete_sequence=["#636EFA"]
        )
        fig.update_layout(
            legend_title_text="Légende",
            showlegend=True,
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠️ Veuillez sélectionner au moins une variable explicative (X) et une variable cible (Y).")
