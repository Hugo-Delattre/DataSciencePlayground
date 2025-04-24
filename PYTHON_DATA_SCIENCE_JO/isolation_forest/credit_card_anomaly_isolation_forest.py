import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("creditcard.csv")

print("Aperçu des données :")
print(df.head())
print(f"Forme des données: {df.shape}")
print(f"Nombre de fraudes: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# Prétraitement
features = df.drop(['Time', 'Class'], axis=1)
labels = df['Class']  # 1 = fraude, 0 = normal

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Isolation Forest (scores d'anomalies)
model = IsolationForest(contamination=0.001, random_state=42)
model.fit(X_scaled)
anomaly_scores = model.decision_function(X_scaled)
anomaly_scores = -anomaly_scores  # Inverser pour que les scores élevés = plus anomaliques

# Prédictions binaires
y_pred = model.predict(X_scaled)
y_pred = [1 if x == -1 else 0 for x in y_pred]

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(labels, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Fraude (pred)'], yticklabels=['Normal', 'Fraude (réel)'])
plt.title("Matrice de confusion - Isolation Forest")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.tight_layout()
plt.show()

# Rapport de classification standard
print("\nRapport de classification :")
print(classification_report(labels, y_pred))

# Métriques supplémentaires
TN, FP, FN, TP = confusion_matrix(labels, y_pred).ravel()
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print("\nMétriques détaillées:")
print(f"Exactitude (Accuracy): {accuracy:.4f}")
print(f"Précision: {precision:.4f}")
print(f"Rappel (Sensibilité): {recall:.4f}")
print(f"Spécificité: {specificity:.4f}")
print(f"Score F1: {f1_score:.4f}")

# Courbe ROC
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(labels, anomaly_scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Courbe Precision-Recall
plt.figure(figsize=(8, 6))
precision_curve, recall_curve, _ = precision_recall_curve(labels, anomaly_scores)
avg_precision = average_precision_score(labels, anomaly_scores)
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'Precision-Recall (AP = {avg_precision:.4f})')
plt.axhline(y=sum(labels)/len(labels), color='red', linestyle='--', label=f'Ligne de base ({sum(labels)/len(labels):.4f})')
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe Precision-Recall')
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()