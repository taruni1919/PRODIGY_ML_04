import numpy as np

import cv2

import os

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Set parameters

img_size = 64 # Size of the input images

batch_size = 32

# Load and preprocess data

def load_data(data_dir):

images = []

labels = []

class_names = os.listdir(data_dir)

for label in class_names:
class_dir = os.path.join(data_dir, label)

for img_name in os.listdir(class_dir): img_path = os.path.join(class_dir, img_name)

img = cv2.imread(img_path)

img = cv2.resize(img, (img_size, img_size))

images.append(img)

labels.append(class_names.index(label))

return np.array(images), np.array(labels)

data_dir = 'path_to_your_data' # Replace with your dataset path X, y = load_data(data_dir)

X = X / 255.0 # Normalize images

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation

train_datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)

train_datagen.fit(X_train)

#Build the model
model = Sequential([

Conv2D(32, (3, 3), activation 'relu', input_shape=(img_size, img_size, 3)),

MaxPooling2D(pool_size=(2, 2)),

Conv2D(64, (3, 3), activation 'relu'),

MaxPooling2D(pool_size=(2, 2)),

Flatten(),

Dense (128, activation='relu'),

Dropout(0.5),

Dense(len(np.unique(y)), activation='softmax') # Number of classes ])

# Compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model

history = model.fit(train_datagen.flow(X_train, y_train, batch_size=batch_size),

validation_data=(X_test, y_test), epochs=10)

# Evaluate the model

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model

model.save('hand_gesture_model.h5')

# Plot training history

plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
