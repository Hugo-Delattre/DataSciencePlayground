
import os
import cv2 # Penser à passer sur le bon environnement : conda activate pneumonie
import numpy as np

print("OpenCV version:", cv2.__version__)

# def load_class_images(folder_path, image_size=(128, 128), max_images=None):
#     X = []
#     images = os.listdir(folder_path)
#     if max_images:
#         images = images[:max_images]
#
#     for file in images:
#         file_path = os.path.join(folder_path, file)
#         img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#         if img is not None:
#             img_resized = cv2.resize(img, image_size)
#             X.append(img_resized.flatten())
#
#     return np.array(X)

# train_normal_path = "D:/.DATASETS/PULMONAIRE/chest_Xray/train/NORMAL/"
# train_pneumonia_path = "D:/.DATASETS/PULMONAIRE/chest_Xray/train/PNEUMONIA"

# X_normal = load_class_images(train_normal_path, image_size=(128, 128), max_images=20)
# X_pneumonia = load_class_images(train_pneumonia_path, image_size=(128, 128), max_images=20)

# print("NORMAL :", X_normal.shape)
# print("PNEUMONIA :", X_pneumonia.shape)
