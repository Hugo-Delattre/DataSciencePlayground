import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class MNISTData:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.X_train = self.X_train.astype('float32') / 255
        self.X_test = self.X_test.astype('float32') / 255
        self.reshape_images()

    def get_shape(self):
        print("X_train shape:", self.X_train.shape)
        print("y_train shape:", self.y_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_test shape:", self.y_test.shape)

    def display_image(self, dataset='train', index=0):
        if dataset == 'train':
            image = self.X_train[index].reshape(28, 28)
            label = self.y_train[index]
        elif dataset == 'test':
            image = self.X_test[index].reshape(28, 28)
            label = self.y_test[index]
        else:
            raise ValueError("Dataset must be 'train' or 'test'")

        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {label}')
        plt.show()

    def display_digit_distribution(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=self.y_train, hue=self.y_train, palette="viridis", legend=False)
        plt.title('Distribution des chiffres dans le jeu de données d\'entraînement')
        plt.xlabel('Chiffre')
        plt.ylabel('Nombre d\'occurrences')
        plt.show()

    def display_mean_images(self):
        mean_images = []
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))

        for digit in range(10):
            mean_img = np.mean(self.X_train[self.y_train == digit], axis=0).reshape(28, 28)
            mean_images.append(mean_img)

            ax = axes[digit // 5, digit % 5]
            ax.imshow(mean_img, cmap="gray")
            ax.set_title(f"Chiffre {digit}")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    def reshape_images(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28 * 28)  # [60000, 784]
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28 * 28)  # [10000, 784]
        print(f"Nouveau format des données : X_train={self.X_train.shape}, X_test={self.X_test.shape}")

    def train_knn(self, k=3):
        print(f"🔍 Entraînement d'un KNN avec k={k}...")
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X_train, self.y_train)

        y_pred = knn.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"✅ Accuracy du modèle KNN (k={k}) : {accuracy:.4f}")

        return knn, y_pred

    def plot_confusion_matrix(self, y_pred):
        conf_matrix = confusion_matrix(self.y_test, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies valeurs")
        plt.title("Matrice de confusion")
        plt.show()


mnist = MNISTData()
mnist.get_shape()
mnist.display_image(dataset='train', index=0)
mnist.display_digit_distribution()
mnist.display_mean_images()
knn_model, y_pred = mnist.train_knn(k=3)
mnist.plot_confusion_matrix(y_pred)