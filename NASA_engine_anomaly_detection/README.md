# Modèle de détection d'anomalies pour la maintenance prédictive d'avions

L'objectif de ce projet est d'analyser les données des capteurs des avions de manière automatique pour:
1. Détecter les anomalies potentielles
2. Prédire la durée de vie restante des composants 
3. Recommander des interventions de maintenance avant les pannes

## Structure du Projet

```
maintenance-predictive-avions/
├── data/                      # Données d'entraînement et de test
├── models/                    # Modèles entraînés et sauvegardés
├── notebooks/                 # Notebooks Jupyter pour l'analyse exploratoire
├── app.py                     # API Flask pour le déploiement
├── Dockerfile                 # Configuration pour le déploiement Docker
├── requirements.txt           # Dépendances Python
├── test_api.py                # Script pour tester l'API
└── README.md                  # Documentation du projet
```

## Modèles Implémentés

1. **Isolation Forest**: Détecte les observations anormales/outliers
2. **Autoencodeur**: Détecte les anomalies basées sur l'erreur de reconstruction
3. **Régression RUL**: Prédit directement la durée de vie résiduelle des composants

## Points Forts du Projet

- Approche multi-modèles pour une détection d'anomalies plus robuste
- Validation croisée pour garantir la fiabilité des résultats
- API déployable pour l'intégration dans les systèmes existants
- Visualisations claires des résultats pour faciliter l'interprétation

## Installation et Déploiement

### Prérequis
- Python 3.8+
- Docker (pour le déploiement containerisé)

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/maintenance-predictive-avions.git
cd maintenance-predictive-avions

# Installer les dépendances
pip install -r requirements.txt
```

### Déploiement Local
```bash
# Lancer l'API
python app.py
```

### Déploiement avec Docker
```bash
# Construire l'image Docker
docker build -t maintenance-predictive:latest .

# Lancer le conteneur
docker run -p 5000:5000 maintenance-predictive:latest
```

## Utilisation de l'API

### Détecter une Anomalie
```bash
curl -X POST http://localhost:5000/api/detect_anomaly \\
  -H "Content-Type: application/json" \\
  -d '{
    "op_setting_1": 20.0,
    "op_setting_2": 0.85,
    "op_setting_3": 100.0,
    "sensor_2": 0.35,[]()
    "sensor_3": 95.0,
    "sensor_4": 42.0,
    "sensor_7": 540.0,
    "sensor_11": 43.0,
    "sensor_12": 5.2,
    "sensor_15": 21.0
  }'
```

### Test de l'API
```bash
# Exécuter le script de test
python test_api.py
```

## Performances des Modèles

- **Isolation Forest**: F1-Score moyen de 0.85 en validation croisée
- **Autoencodeur**: AUC-ROC de 0.92 sur les données de test
- **Prédiction RUL**: MAE de 15.8 cycles sur les données de test
