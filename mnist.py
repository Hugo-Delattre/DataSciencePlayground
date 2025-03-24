import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

class MNISTDataset:
    def __init__(self):
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.load_data()

    def load_data(self):
        (self.train_X, self.train_y), (self.test_X, self.test_y) = mnist.load_data()
        self.print_shapes()

    def print_shapes(self):
        print('X_train shape: ' + str(self.train_X.shape))
        print('Y_train shape: ' + str(self.train_y.shape))
        print('X_test shape:  ' + str(self.test_X.shape))
        print('Y_test shape:  ' + str(self.test_y.shape))

    def display_samples(self, num_samples=9):
        for i in range(min(num_samples, 9)):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.train_X[i], cmap=plt.get_cmap('gray'))
        plt.show()

    def get_data(self):
        return self.train_X, self.train_y, self.test_X, self.test_y

    def display_digit_counts(self):
        train_unique, train_counts = np.unique(self.train_y, return_counts=True)
        test_unique, test_counts = np.unique(self.test_y, return_counts=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        bar_width = 0.35

        x = np.arange(len(train_unique))

        train_bars = ax.bar(x - bar_width / 2, train_counts, bar_width, label='Train')
        test_bars = ax.bar(x + bar_width / 2, test_counts, bar_width, label='Test')

        ax.set_xlabel('Chiffre')
        ax.set_ylabel('Nombre')
        ax.set_title("Nombre d'exemplaire de chaque chiffre dans le dataset")
        ax.set_xticks(x)
        ax.set_xticklabels(train_unique)
        ax.legend()

        for i, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
            ax.text(i - bar_width / 2, train_count + 50, str(train_count), ha='center')
            ax.text(i + bar_width / 2, test_count + 50, str(test_count), ha='center')

        plt.tight_layout()
        plt.show()

        print("Digit | Train Count | Test Count")
        print("-" * 30)
        for digit, train_count, test_count in zip(train_unique, train_counts, test_counts):
            print(f"{digit:5d} | {train_count:11d} | {test_count:10d}")

    def display_specific_image(self, index, dataset="train"):
        if dataset.lower() == "train":
            image = self.train_X[index]
            label = self.train_y[index]
        elif dataset.lower() == "test":
            image = self.test_X[index]
            label = self.test_y[index]
        else:
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Digit: {label} (from {dataset} set)")
        plt.tight_layout()
        plt.show()

    def display_digit_means(self):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for digit in range(10):
            indices = np.where(self.train_y == digit)[0]
            digit_image = self.train_X[indices]
            mean_image = np.mean(digit_image, axis=0)
            axes[digit].imshow(mean_image, cmap='gray')

        plt.tight_layout()
        plt.show()

    def flatten_images(self):
        num_features = self.train_X.shape[1] * self.train_X.shape[2]

        # [n, 28, 28] -> [n, 784]
        self.train_X_flat = self.train_X.reshape(self.train_X.shape[0], num_features)

        # [n, 28, 28] -> [n, 784]
        self.test_X_flat = self.test_X.reshape(self.test_X.shape[0], num_features)

        # Print the new shapes
        print(f'Flattened X_train shape: {self.train_X_flat.shape}')
        print(f'Flattened X_test shape: {self.test_X_flat.shape}')

        return self.train_X_flat, self.test_X_flat

    def get_flattened_data(self):
        if not hasattr(self, 'train_X_flat') or self.train_X_flat is None:
            self.flatten_images()
        return self.train_X_flat, self.train_y, self.test_X_flat, self.test_y

mnist_data = MNISTDataset()

mnist_data.display_samples()

train_X, train_y, test_X, test_y = mnist_data.get_data()

mnist_data.display_digit_counts()

mnist_data.display_digit_means()

mnist_data.display_specific_image(0, "train")

train_X_flat, train_y, test_X_flat, test_y = mnist_data.get_flattened_data()