import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)
X_normal = 0.3 * rng.randn(100, 2)
X_normal = np.r_[X_normal + 2, X_normal - 2]

X_anomalies = rng.uniform(low=-4, high=4, size=(20, 2))

X = np.r_[X_normal, X_anomalies]

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X)

y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], c='blue', label='Normal')
plt.scatter(X[y_pred == -1][:, 0], X[y_pred == -1][:, 1], c='red', label='Anomalie')
plt.title("Détection d'anomalies avec Isolation Forest")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
