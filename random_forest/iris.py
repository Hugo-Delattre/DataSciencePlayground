import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Évaluation du modèle
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle: {accuracy:.4f}")

print("\nRapport de classification:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Matrice de confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.title("Importance des caractéristiques")
plt.tight_layout()
plt.show()

# Sauvegarder le modèle et le scaler pour le déploiement avec joblib
dump(clf, 'model.joblib')
dump(scaler, 'scaler.joblib')

# Fonction de prédiction pour le déploiement
def predict(data):
    """
    Fonction pour faire des prédictions avec le modèle entraîné

    Args:
        data (array): Les caractéristiques à prédire (4 valeurs pour Iris)

    Returns:
        tuple: (classe prédite (index), nom de la classe, probabilités)
    """
    # Charger le modèle et le scaler avec joblib
    model = load('model.joblib')
    scaler_loaded = load('scaler.joblib')

    # Prétraiter les données
    data_scaled = scaler_loaded.transform([data])

    # Prédire la classe
    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0]

    return prediction, target_names[prediction], proba

# Exemple d'utilisation de la fonction de prédiction
sample_data = [5.1, 3.5, 1.4, 0.2]  # Une fleur Setosa
pred_idx, pred_name, probas = predict(sample_data)
print(f"\nExemple de prédiction:")
print(f"Données: {sample_data}")
print(f"Classe prédite: {pred_name} (index {pred_idx})")
print(f"Probabilités: {probas}")