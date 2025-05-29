import streamlit as st
import pandas as pd
import joblib

# Charger les modèles
model_rf = joblib.load("model_rf.joblib")
model_kmeans = joblib.load("model_kmeans.joblib")
scaler = joblib.load("scaler.joblib")

# Liste des features
features = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

st.title("🧾 Détection de faux billets")

st.write("Cette application utilise un modèle supervisé (Random Forest) et un modèle non supervisé (KMeans) pour détecter les faux billets.")

# Partie 1 : Saisie manuelle d’un billet
st.subheader("Entrez les caractéristiques du billet :")
inputs = {}
for feature in features:
    inputs[feature] = st.number_input(f"{feature}", step=0.1)

billet_df = pd.DataFrame([inputs])

if st.button("Analyser le billet"):
    pred_rf = model_rf.predict(billet_df)[0]
    proba_rf = model_rf.predict_proba(billet_df)[0][1]
    billet_scaled = scaler.transform(billet_df)
    cluster = model_kmeans.predict(billet_scaled)[0]

    st.markdown("### Résultat de la prédiction :")
    if pred_rf:
        st.success(f"✅ [Random Forest] Ce billet est AUTHENTIQUE (confiance : {proba_rf:.2%})")
    else:
        st.error(f"❌ [Random Forest] Ce billet est FAUX (confiance : {(1 - proba_rf):.2%})")

    st.info(f"🔎 [KMeans] Cluster assigné : {cluster}")
    
    # Légende explicative des clusters
    st.markdown("""
    ---
    **Légende des clusters KMeans :**
    - Cluster 0 : contient environ 98.6% de billets authentiques (is_genuine proche de 1)
    - Cluster 1 : contient environ 1.85% de billets authentiques, donc majoritairement des billets faux
    """)

# Partie 2 : Upload d’un fichier CSV
st.subheader("Ou importez un fichier CSV avec plusieurs billets :")
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)

    # Vérifier que toutes les colonnes nécessaires sont présentes
    if all(feature in df_upload.columns for feature in features):
        billets_features = df_upload[features]

        # Prédictions Random Forest
        preds_rf = model_rf.predict(billets_features)
        probas_rf = model_rf.predict_proba(billets_features)[:, 1]

        # KMeans clustering
        billets_scaled = scaler.transform(billets_features)
        clusters = model_kmeans.predict(billets_scaled)

        # Ajouter les résultats au dataframe
        df_upload['Prediction_RF'] = preds_rf
        df_upload['Proba_RF'] = probas_rf
        df_upload['Cluster_KMeans'] = clusters

        st.markdown("### Résultats pour le fichier importé :")
        st.dataframe(df_upload)

        # Légende explicative des clusters sous le tableau
        st.markdown("""
        ---
        **Légende des clusters KMeans :**
        - Cluster 0 : contient environ 98.6% de billets authentiques
        - Cluster 1 : contient environ 1.85% de billets authentiques, donc majoritairement des billets faux
        """)

        # Optionnel : bouton pour télécharger les résultats
        csv = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les résultats", data=csv, file_name="resultats_detection.csv", mime="text/csv")

    else:
        st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {features}")
