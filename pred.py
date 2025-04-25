import streamlit as st
import pandas as pd
import plotly.express as px
from math import sqrt
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# PDF option (need fpdf)
try:
    from fpdf import FPDF
    has_fpdf = True
except ImportError:
    has_fpdf = False

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
            rmse = sqrt(mean_squared_error(y_test, y_pred))  # Remplace squared=False

            st.subheader("📊 Résultats du Modèle")
            st.write(f"**R² Score** : {r2:.3f}")
            st.write(f"**MAE** (Erreur Absolue Moyenne) : {mae:.3f}")
            st.write(f"**RMSE** (Erreur Quadratique Moyenne) : {rmse:.3f}")

            # 📐 Affichage de l'équation (pour Régression Linéaire uniquement)
            if model_choice == "Régression Linéaire":
                coefficients = model.coef_
                intercept = model.intercept_
                equation = f"{y_col} = " + " + ".join(
                    [f"{coef:.3f} * {col}" for coef, col in zip(coefficients, x_cols)]
                ) + f" + {intercept:.3f}"

                st.markdown("### 📐 Équation de la Régression Linéaire")
                st.code(equation, language="python")

                # 📄 Télécharger l'équation en TXT
                equation_txt = f"Équation de la Régression Linéaire\n\n{equation}"
                txt_buffer = io.StringIO()
                txt_buffer.write(equation_txt)

                st.download_button(
                    label="📄 Télécharger l'équation (.txt)",
                    data=txt_buffer.getvalue(),
                    file_name="equation_regression.txt",
                    mime="text/plain"
                )

                # 📄 Télécharger l'équation en PDF (si disponible)
                if has_fpdf:
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, equation_txt)
                    pdf_buffer = io.BytesIO()
                    pdf.output(pdf_buffer)
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="📄 Télécharger l'équation (.pdf)",
                        data=pdf_buffer,
                        file_name="equation_regression.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.info("📎 Pour activer le téléchargement en PDF, ajoute `fpdf` dans ton fichier requirements.txt.")

            # 📉 Visualisation dynamique avec Plotly
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
                labels={"Valeur Réelle": "CR Réel", "Valeur Prédite": "CR Prédit"}
            )
            fig.update_layout(
                legend_title_text="Légende",
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
