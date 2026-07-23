"""
Entraînement du modèle de détection de fraude bancaire.

Jeu de données réel : Bank_transaction_scenario1.csv
Colonnes sources : ID Clients ; Numero de compte ; Identifiant operation ;
Type de transaction ; Status operation ; Localisation ; Date ; Montant ; Target
Target : Normal / Suspect / Fraude
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# -----------------------------------------------------------------
# 1. Chargement des données
# -----------------------------------------------------------------
df = pd.read_csv("data/transactions.csv", sep=";")

# -----------------------------------------------------------------
# 2. Feature engineering
# -----------------------------------------------------------------
# Date -> composantes temporelles exploitables par le modèle
df["Date"] = pd.to_datetime(df["Date"])
df["Heure"] = df["Date"].dt.hour
df["JourSemaine"] = df["Date"].dt.dayofweek  # 0 = lundi
df["Weekend"] = df["JourSemaine"].isin([5, 6]).astype(int)

# Regrouper les localisations rares (long tail) pour éviter le surapprentissage
seuil_min = 20
freq_localisation = df["Localisation"].value_counts()
localisations_frequentes = freq_localisation[freq_localisation >= seuil_min].index
df["Localisation_grp"] = df["Localisation"].where(
    df["Localisation"].isin(localisations_frequentes), "Autre"
)

# Fréquence historique de transactions par client (proxy simple de comportement habituel)
freq_client = df["ID Clients"].value_counts()
df["Freq_client"] = df["ID Clients"].map(freq_client)

# Encodage one-hot des variables catégorielles
categorical_cols = ["Type de transaction", "Status operation", "Localisation_grp"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)

feature_cols = (
    ["Montant", "Heure", "JourSemaine", "Weekend", "Freq_client"]
    + [c for c in df_encoded.columns if c.startswith(tuple(categorical_cols))]
)

X = df_encoded[feature_cols]
y_raw = df["Target"]

# -----------------------------------------------------------------
# 3. Encodage de la cible (Normal / Suspect / Fraude)
# -----------------------------------------------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print("Classes détectées :", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
print(y_raw.value_counts())

# -----------------------------------------------------------------
# 4. Normalisation
# -----------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------
# 5. Split train/test (stratifié car classes déséquilibrées)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 6. Entraînement
# -----------------------------------------------------------------
# Point d'attention pédagogique : le jeu de données est déséquilibré
# (Normal 76 %, Suspect 20 %, Fraude 4 %). On utilise class_weight="balanced"
# et on privilégie precision/recall/F1 plutôt que l'accuracy seule.
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# 7. Évaluation
# -----------------------------------------------------------------
y_pred = model.predict(X_test)
print("\n=== Rapport de classification ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("=== Matrice de confusion ===")
print(confusion_matrix(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Importance des variables (top 10) ===")
print(importances.head(10))

# -----------------------------------------------------------------
# 8. Sauvegarde du modèle, du scaler, de l'encodeur et des métadonnées
# -----------------------------------------------------------------
joblib.dump(model, "model/fraud_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

metadata = {
    "feature_cols": feature_cols,
    "categorical_cols": categorical_cols,
    "localisations_frequentes": sorted(localisations_frequentes.tolist()),
    "types_transaction": sorted(df["Type de transaction"].unique().tolist()),
    "status_operation": sorted(df["Status operation"].unique().tolist()),
    "classes": label_encoder.classes_.tolist(),
}
with open("model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nModèle, scaler, encodeur et métadonnées sauvegardés dans model/")
