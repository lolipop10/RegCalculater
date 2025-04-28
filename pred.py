import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Configuration de la page Streamlit
st.set_page_config(page_title="📈 Modélisation de Régression", layout="wide")

# Titre
st.title("📊 Application de Modélisation - Régression & Random Forest")

# Upload du fichier
uploaded_file = st.file_uploader("📤 Charger un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("🗂️ Aperçu du fichier chargé :")
    st.dataframe(df.head())

    # --- Zone Filtrage ---
    st.sidebar.header("🔎 Filtrer les Données")
    filtre_colonne = st.sidebar.selectbox("Sélectionner une colonne pour filtrer :", ["-- Aucun filtre --"] + df.columns.tolist())
    
    if filtre_colonne != "-- Aucun filtre --":
        unique_values = df[filtre_colonne].dropna().unique().tolist()
        selected_value = st.sidebar.selectbox(f"Sélectionner une valeur :", unique_values)
        df = df[df[filtre_colonne] == selected_value]

    # --- Zone Modélisation ---
    st.sidebar.header("⚙️ Paramètres du Modèle")

    colonnes = df.columns.tolist()
    x_cols = st.sidebar.multiselect("Variables explicatives (X)", colonnes)
    y_col = st.sidebar.selectbox("Variable cible à prédire (Y)", colonnes)

    model_choice = st.sidebar.selectbox("Choisir le Modèle :", ["Régression Linéaire", "Random Forest"])
    split_option = st.sidebar.radio("Séparer les données en train/test ?", ["Oui", "Non (utiliser tout)"])

    if x_cols and y_col:
        # Nettoyage
        data = df[x_cols + [y_col]].dropna()
        X = data[x_cols]
        y = data[y_col]

        # Séparation train/test
        if split_option == "Oui":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Modèle
        if model_choice == "Régression Linéaire":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Entraînement
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Résultats
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        st.subheader("📈 Résultats du Modèle")
        st.metric("R² (Coefficient de détermination)", f"{r2:.3f}")
        st.metric("MAE (Erreur Absolue Moyenne)", f"{mae:.3f}")
        st.metric("RMSE (Erreur Quadratique Moyenne)", f"{rmse:.3f}")

        # Graphique dynamique Plotly
        st.subheader("📉 Visualisation : Réel vs Prédit")

        results_df = pd.DataFrame({
            "Valeur Réelle": y_test,
            "Valeur Prédite": y_pred
        })

        fig = px.scatter(
            results_df,
            x="Valeur Réelle",
            y="Valeur Prédite",
            title="Réel vs Prédit",
            labels={"Valeur Réelle": "Y Réel", "Valeur Prédite": "Y Prédit"},
            trendline="ols"
        )

        fig.update_layout(
            legend_title_text="Légende",
            showlegend=True,
            width=800,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("📥 Veuillez charger un fichier Excel pour commencer.")
