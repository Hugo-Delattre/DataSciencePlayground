import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mnist_loader import MNISTDataset

mnist_data = MNISTDataset()
train_X_flat, train_y, test_X_flat, test_y = mnist_data.get_flattened_data()

# Choisir un modèle (ici, une régression logistique)
model = LogisticRegression(max_iter=1000)

# Entraîner le modèle
print("Entraînement du modèle...")
model.fit(train_X_flat, train_y)

# Prédictions
predictions = model.predict(test_X_flat)

# Évaluation des performances
accuracy = accuracy_score(test_y, predictions)
print(f"Précision du modèle : {accuracy:.4f}")
