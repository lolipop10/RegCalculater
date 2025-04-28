import streamlit as st
import pandas as pd
import plotly.express as px
from math import sqrt
from itertools import product
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Modélisation Régression", layout="wide")
st.title("📈 Application de Modélisation de Régression")

uploaded_file = st.file_uploader("📤 Charger un fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Aperçu du fichier chargé :")
    st.dataframe(df.head())

    st.sidebar.header("🧰 Filtrage des données")
    filtre_colonne = st.sidebar.selectbox("Choisir une colonne pour filtrer :", options=["-- Aucun filtre --"] + df.columns.tolist())

    if filtre_colonne != "-- Aucun filtre --":
        unique_values = df[filtre_colonne].dropna().unique().tolist()
        selected_value = st.sidebar.selectbox(f"Valeur de '{filtre_colonne}' à filtrer :", unique_values)
        df = df[df[filtre_colonne] == selected_value]

    colonnes = df.columns.tolist()
    st.sidebar.header("🔧 Sélection des variables")
    x_cols = st.sidebar.multiselect("Variables explicatives (X)", colonnes)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    y_col = st.sidebar.selectbox("Variable à prédire (Y)", numeric_columns)

    if x_cols and y_col:
        data = df[x_cols + [y_col]].dropna()
        if data.empty:
            st.warning("⚠️ Aucune donnée disponible après nettoyage.")
        else:
            X = data[x_cols]
            y = data[y_col]

            if not pd.api.types.is_numeric_dtype(y):
                st.error(f"❌ La variable à prédire '{y_col}' n'est pas numérique.")
            else:
                model_choice = st.sidebar.selectbox("Choisir le modèle :", ["Régression Linéaire", "Random Forest"])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                if X_train.empty or y_train.empty:
                    st.warning("⚠️ Pas assez de données pour entraîner le modèle.")
                else:
                    if model_choice == "Régression Linéaire":
                        model = LinearRegression()
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = sqrt(mean_squared_error(y_test, y_pred))

                    st.subheader("📊 Résultats du Modèle")
                    st.write(f"**R² Score** : {r2:.3f}")
                    st.write(f"**MAE** (Erreur Absolue Moyenne) : {mae:.3f}")
                    st.write(f"**RMSE** (Erreur Quadratique Moyenne) : {rmse:.3f}")

                    st.subheader("📐 Équation du Modèle / Importance des Variables")

                    if model_choice == "Régression Linéaire":
                        try:
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
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage de l'équation : {e}")

                    else:
                        importances = model.feature_importances_
                        importance_df = pd.DataFrame({
                            "Variable": x_cols,
                            "Importance": importances
                        }).sort_values(by="Importance", ascending=False)

                        st.dataframe(importance_df)

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
                    fig.update_layout(showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("🎯 Illustration du Modèle Entraîné")

                    if len(x_cols) == 1:
                        var_x = x_cols[0]
                        x_range = pd.Series(sorted(X[var_x].unique()))
                        X_vis = pd.DataFrame({col: X[col].mean() for col in x_cols}, index=x_range.index)
                        X_vis[var_x] = x_range.values
                        X_vis = X_vis[x_cols]  # Assure bon ordre des colonnes
                        y_vis_pred = model.predict(X_vis)

                        pred_df = pd.DataFrame({
                            var_x: x_range,
                            f"Prédiction de {y_col}": y_vis_pred
                        })

                        real_df = pd.DataFrame({
                            var_x: X[var_x],
                            y_col: y
                        })

                        fig2 = px.scatter(real_df, x=var_x, y=y_col, opacity=0.6, labels={var_x: var_x, y_col: f"{y_col} réel"}, title=f"{y_col} réel vs prédiction en fonction de {var_x}")
                        fig2.add_scatter(x=pred_df[var_x], y=pred_df[f"Prédiction de {y_col}"], mode='lines', name='Prédiction', line=dict(color='red'))

                        st.plotly_chart(fig2, use_container_width=True)

                    elif len(x_cols) == 2:
                        x1_vals = np.linspace(X[x_cols[0]].min(), X[x_cols[0]].max(), 30)
                        x2_vals = np.linspace(X[x_cols[1]].min(), X[x_cols[1]].max(), 30)
                        grid = pd.DataFrame(product(x1_vals, x2_vals), columns=x_cols)
                        preds = model.predict(grid)
                        grid[y_col] = preds

                        fig3d = px.scatter_3d(
                            grid, x=x_cols[0], y=x_cols[1], z=y_col,
                            color=y_col, opacity=0.7,
                            title=f"Prédictions de {y_col} selon {x_cols[0]} et {x_cols[1]}"
                        )
                        st.plotly_chart(fig3d, use_container_width=True)

                    else:
                        st.info("ℹ️ L'illustration du modèle est disponible uniquement pour 1 ou 2 variables explicatives.")
