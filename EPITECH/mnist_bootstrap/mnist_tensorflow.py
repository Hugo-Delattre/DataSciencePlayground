import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MNISTData:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.X_train = self.X_train.astype('float32') / 255  # Normalisation
        self.X_test = self.X_test.astype('float32') / 255
        self.reshape_images()

    def reshape_images(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28 * 28)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28 * 28)

    def train_mlp(self, epochs=10, batch_size=32):
        print("🚀 Entraînement du réseau de neurones...")

        # Définition
        model = keras.Sequential([
            layers.Dense(128, activation="relu", input_shape=(784,)),  # Couche cachée 1
            layers.Dense(64, activation="relu"),  # Couche cachée 2
            layers.Dense(10, activation="softmax")  # Couche de sortie (10 classes)
        ])

        # Compilation
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        # Entraînement
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test))

        # Évaluation
        test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"✅ Accuracy du modèle MLP : {test_acc:.4f}")

        return model


mnist = MNISTData()
mlp_model = mnist.train_mlp(epochs=10, batch_size=32)
