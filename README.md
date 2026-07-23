# Détection de fraude bancaire — Streamlit

Application de détection de fraude bancaire entraînée sur des données réelles
de transactions (Sénégal), avec interface Streamlit pour l'analyse unitaire
ou par lot (CSV).

## Données

`data/transactions.csv` (séparateur `;`) contient les colonnes :

| Colonne | Description |
|---|---|
| ID Clients | Identifiant du client |
| Numero de compte | Numéro de compte bancaire |
| Identifiant operation | Identifiant unique de l'opération |
| Type de transaction | ATM, Paiement en ligne, Paiement électronique |
| Status operation | Validé, Échoué, En attente |
| Localisation | Ville/région de l'opération |
| Date | Horodatage de la transaction |
| Montant | Montant en FCFA |
| Target | Normal / Suspect / Fraude (variable à prédire) |

Répartition des classes : ~76 % Normal, ~20 % Suspect, ~4 % Fraude
(fortement déséquilibré, comme la plupart des jeux de données de fraude).

## Structure du projet

```
fraude-detection-streamlit/
│
├── data/
│   └── transactions.csv       # Jeu de données réel
│
├── model/
│   ├── train_model.py         # Script d'entraînement
│   ├── fraud_model.pkl        # Modèle sérialisé (généré)
│   ├── scaler.pkl             # Normaliseur (généré)
│   ├── label_encoder.pkl      # Encodeur de la cible (généré)
│   └── metadata.json          # Colonnes/catégories utilisées (généré)
│
├── app.py                     # Application Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Entraînement du modèle

```bash
python model/train_model.py
```

Cela génère `fraud_model.pkl`, `scaler.pkl`, `label_encoder.pkl` et
`metadata.json` dans `model/`. Résultats obtenus sur ce jeu de données
(RandomForest, `class_weight="balanced"`) :

- Accuracy globale : ~90 %
- F1-score classe Fraude : ~0.70 (recall 0.75, precision 0.65)
- F1-score classe Suspect : ~0.78
- F1-score classe Normal : ~0.94

Les variables les plus discriminantes sont le **Montant**, la **fréquence
historique du client**, le **statut de l'opération** et l'**heure** de la
transaction.

## Lancer l'application en local

```bash
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Déploiement sur Streamlit Community Cloud

1. Publier le projet sur GitHub (le `.gitignore` exclut déjà `venv/` et
   les fichiers non nécessaires — vérifier que `model/*.pkl` est bien
   inclus dans le dépôt, sinon le modèle ne se chargera pas sur le cloud).
2. Aller sur [share.streamlit.io](https://share.streamlit.io) et se
   connecter avec GitHub.
3. Cliquer sur **New app**, choisir le dépôt, la branche `main` et le
   fichier `app.py`.
4. Cliquer sur **Deploy**.

L'application sera accessible sur une URL du type
`https://<nom-app>.streamlit.app`.

## Notes de sécurité

Ce jeu de données contient des identifiants clients et numéros de compte.
Avant toute publication sur un dépôt GitHub public, anonymiser ou retirer
ces colonnes, ou héberger le CSV dans un stockage privé et le charger via
`st.secrets` plutôt que de le committer.

## Pour aller plus loin

- Ajouter une visualisation SHAP pour expliquer chaque prédiction
- Ajouter un onglet de suivi de dérive des données (data drift) dans le temps
- Comparer plusieurs modèles (RandomForest, XGBoost, Isolation Forest)
- Ajouter une authentification utilisateur (`streamlit-authenticator`)
