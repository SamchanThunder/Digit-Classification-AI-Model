from email.mime import image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#Retrieve data and split them into training vs test. 
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

#Scale from 0-255 to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy")
print("Training Data")
model.fit(train_images, train_labels, epochs = 5, batch_size = 32)

tf.saved_model.save(model, '')