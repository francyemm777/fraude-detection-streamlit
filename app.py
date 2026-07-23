import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Configuration de la page ---
st.set_page_config(
    page_title="Détection de Fraude Bancaire",
    page_icon="🏦",
    layout="wide"
)

# --- Chargement du modèle, du scaler, de l'encodeur et des métadonnées (mis en cache) ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    with open("model/metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, label_encoder, metadata

model, scaler, label_encoder, metadata = load_artifacts()
FEATURE_COLS = metadata["feature_cols"]
LOCALISATIONS = metadata["localisations_frequentes"]
TYPES_TRANSACTION = metadata["types_transaction"]
STATUS_OPERATION = metadata["status_operation"]

# --- En-tête ---
st.title("🏦 Système de Détection de Fraude Bancaire")
st.markdown("Analysez une transaction ou un lot de transactions pour détecter un risque de fraude.")

# --- Menu latéral ---
mode = st.sidebar.radio("Mode d'analyse", ["Transaction unique", "Fichier CSV (lot)"])


def build_features(montant, heure, jour_semaine, localisation, type_transaction, status_operation, freq_client):
    """Construit le vecteur de features dans le même ordre que l'entraînement."""
    row = {col: 0 for col in FEATURE_COLS}
    row["Montant"] = montant
    row["Heure"] = heure
    row["JourSemaine"] = jour_semaine
    row["Weekend"] = 1 if jour_semaine >= 5 else 0
    row["Freq_client"] = freq_client

    loc_grp = localisation if localisation in LOCALISATIONS else "Autre"
    col_loc = f"Localisation_grp_{loc_grp}"
    if col_loc in row:
        row[col_loc] = 1

    col_type = f"Type de transaction_{type_transaction}"
    if col_type in row:
        row[col_type] = 1

    col_status = f"Status operation_{status_operation}"
    if col_status in row:
        row[col_status] = 1

    return pd.DataFrame([row])[FEATURE_COLS]


def afficher_resultat(prediction_label, probas):
    """Affiche le résultat coloré selon la classe prédite."""
    proba_dict = dict(zip(label_encoder.classes_, probas))
    if prediction_label == "Fraude":
        st.error(f"🚨 **Fraude probable** — Probabilité : {proba_dict['Fraude']:.1%}")
    elif prediction_label == "Suspect":
        st.warning(f"⚠️ **Transaction suspecte** — Probabilité : {proba_dict['Suspect']:.1%}")
    else:
        st.success(f"✅ **Transaction normale** — Probabilité : {proba_dict['Normal']:.1%}")

    cols = st.columns(3)
    for c, classe in zip(cols, label_encoder.classes_):
        c.metric(classe, f"{proba_dict[classe]:.1%}")


# ============================
# MODE 1 : Transaction unique
# ============================
if mode == "Transaction unique":
    st.subheader("Saisie manuelle d'une transaction")
    col1, col2 = st.columns(2)

    with col1:
        montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
        heure = st.slider("Heure de la transaction (0-23)", 0, 23, 12)
        jour_semaine = st.selectbox(
            "Jour de la semaine",
            options=list(range(7)),
            format_func=lambda x: ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"][x],
        )
        freq_client = st.number_input(
            "Nombre d'opérations historiques du client", min_value=1, value=5,
            help="Proxy du comportement habituel du client : plus il a d'opérations connues, plus son profil est établi."
        )

    with col2:
        localisation = st.selectbox("Localisation", options=LOCALISATIONS + ["Autre"])
        type_transaction = st.selectbox("Type de transaction", options=TYPES_TRANSACTION)
        status_operation = st.selectbox("Statut de l'opération", options=STATUS_OPERATION)

    if st.button("Analyser la transaction", type="primary"):
        X_input = build_features(montant, heure, jour_semaine, localisation, type_transaction, status_operation, freq_client)
        X_scaled = scaler.transform(X_input)
        pred_idx = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        prediction_label = label_encoder.inverse_transform([pred_idx])[0]

        st.divider()
        afficher_resultat(prediction_label, proba)

# ============================
# MODE 2 : Fichier CSV (lot)
# ============================
else:
    st.subheader("Analyse par lot (fichier CSV)")
    st.caption("Le fichier doit respecter le même format que les données d'entraînement : "
               "colonnes séparées par ';' — ID Clients, Numero de compte, Identifiant operation, "
               "Type de transaction, Status operation, Localisation, Date, Montant.")

    fichier = st.file_uploader("Déposez un fichier CSV de transactions", type=["csv"])

    if fichier is not None:
        df = pd.read_csv(fichier, sep=";")
        st.write("Aperçu des données :", df.head())

        if st.button("Lancer l'analyse du lot"):
            df_work = df.copy()
            df_work["Date"] = pd.to_datetime(df_work["Date"])
            df_work["Heure"] = df_work["Date"].dt.hour
            df_work["JourSemaine"] = df_work["Date"].dt.dayofweek
            df_work["Weekend"] = df_work["JourSemaine"].isin([5, 6]).astype(int)
            df_work["Localisation_grp"] = df_work["Localisation"].where(
                df_work["Localisation"].isin(LOCALISATIONS), "Autre"
            )
            freq_client_map = df_work["ID Clients"].value_counts()
            df_work["Freq_client"] = df_work["ID Clients"].map(freq_client_map)

            df_encoded = pd.get_dummies(
                df_work, columns=["Type de transaction", "Status operation", "Localisation_grp"],
                prefix=["Type de transaction", "Status operation", "Localisation_grp"],
            )
            for col in FEATURE_COLS:
                if col not in df_encoded:
                    df_encoded[col] = 0
            X = df_encoded[FEATURE_COLS]

            X_scaled = scaler.transform(X)
            pred_idx = model.predict(X_scaled)
            probas = model.predict_proba(X_scaled)

            df["prediction"] = label_encoder.inverse_transform(pred_idx)
            for i, classe in enumerate(label_encoder.classes_):
                df[f"proba_{classe}"] = probas[:, i]

            nb_fraudes = (df["prediction"] == "Fraude").sum()
            nb_suspects = (df["prediction"] == "Suspect").sum()
            st.warning(f"**{nb_fraudes}** fraude(s) et **{nb_suspects}** transaction(s) suspecte(s) "
                       f"détectée(s) sur {len(df)}.")

            def couleur_ligne(row):
                if row["prediction"] == "Fraude":
                    return ["background-color: #ffb3b3"] * len(row)
                elif row["prediction"] == "Suspect":
                    return ["background-color: #ffe0b3"] * len(row)
                return [""] * len(row)

            st.dataframe(df.style.apply(couleur_ligne, axis=1))

            csv_export = df.to_csv(index=False, sep=";").encode("utf-8")
            st.download_button("Télécharger les résultats", csv_export, "resultats_analyse.csv", "text/csv")

# --- Pied de page ---
st.sidebar.markdown("---")
st.sidebar.caption("Projet pédagogique — Détection de fraude bancaire par IA")
