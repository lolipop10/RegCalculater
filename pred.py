import streamlit as st
import pandas as pd
import plotly.express as px
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Modélisation Régression", layout="wide")

# Titre de l'application
st.title("📈 Application de Modélisation de Régression")

# Upload du fichier Excel
uploaded_file = st.file_uploader("📤 Charger un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Aperçu du fichier chargé :")
    st.dataframe(df.head())

    # 🔎 Zone de filtre
    st.sidebar.header("🧰 Filtrage des données")
    filtre_colonne = st.sidebar.selectbox("Choisir une colonne pour filtrer :", options=["-- Aucun filtre --"] + df.columns.tolist())

    if filtre_colonne != "-- Aucun filtre --":
        unique_values = df[filtre_colonne].dropna().unique().tolist()
        selected_value = st.sidebar.selectbox(f"Valeur de '{filtre_colonne}' à filtrer :", unique_values)
        df = df[df[filtre_colonne] == selected_value]

    # Choix des colonnes pour X et Y
    colonnes = df.columns.tolist()
    st.sidebar.header("🔧 Sélection des variables")

    x_cols = st.sidebar.multiselect("Variables explicatives (X)", colonnes)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    y_col = st.sidebar.selectbox("Variable à prédire (Y)", numeric_columns)

    if x_cols and y_col:
        # Nettoyage des données
        data = df[x_cols + [y_col]].dropna()
        X = data[x_cols]
        y = data[y_col]

        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"❌ La variable à prédire '{y_col}' n'est pas numérique.")
        else:
            # Choix du modèle
            model_choice = st.sidebar.selectbox("Choisir le modèle :", ["Régression Linéaire", "Random Forest"])

            # Split des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Création du modèle
            if model_choice == "Régression Linéaire":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Résultats
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = sqrt(mean_squared_error(y_test, y_pred))

            st.subheader("📊 Résultats du Modèle")
            st.write(f"**R² Score** : {r2:.3f}")
            st.write(f"**MAE** (Erreur Absolue Moyenne) : {mae:.3f}")
            st.write(f"**RMSE** (Erreur Quadratique Moyenne) : {rmse:.3f}")

            # 🔍 Affichage de l'équation ou de l'importance
            st.subheader("📐 Équation du Modèle / Importance des Variables")

            if model_choice == "Régression Linéaire":
                terms = []
                latex_terms = []
                for i, col in enumerate(x_cols):
                    coef = model.coef_[i]
                    if abs(coef) < 1e-4:
                        formatted_coef = f"{coef:.2e}"
                        latex_coef = formatted_coef.replace("e", "\\times 10^{") + "}"
                    else:
                        formatted_coef = f"{coef:.3f}"
                        latex_coef = formatted_coef
                    terms.append(f"{formatted_coef} × {col}")
                    latex_terms.append(f"{latex_coef} \\times {col}")

                intercept = model.intercept_
                equation = f"{y_col} = " + " + ".join(terms) + f" + {intercept:.3f}"
                st.code(equation, language="markdown")

                latex_eq = f"{y_col} = " + " + ".join(latex_terms) + f" + {intercept:.3f}"
                st.latex(latex_eq)

            else:
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Variable": x_cols,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)

                st.dataframe(importance_df)

            # Visualisation dynamique avec Plotly
            st.subheader("📉 Graphique dynamique des prédictions")

            results_df = pd.DataFrame({
                "Valeur Réelle": y_test,
                "Valeur Prédite": y_pred
            })

            fig = px.scatter(
                results_df,
                x="Valeur Réelle",
                y="Valeur Prédite",
                title="Réel vs Prédit",
                labels={"Valeur Réelle": "CR Réel", "Valeur Prédite": "CR Prédit"},
                trendline="ols"
            )
            fig.update_layout(
                legend_title_text="Légende",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
