import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from layers import FRMap, ReOrth, ProjMap, ProjPooling, OrthMap
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputLayer, Dense, Flatten, AveragePooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

new_X = []
for x in X_train:
    q, r = np.linalg.qr(x / 255)
    new_X.append(q)
X_train = np.stack(new_X)

new_X = []
for x in X_test:
    q, r = np.linalg.qr(x / 255)
    new_X.append(q)
X_test = np.stack(new_X)

plt.imshow(X_train[0], cmap="gray")
plt.show()

batch_size = 64
#           (d0, q)     first and last dimensions are data dimension: (d,q).
input_dim = (28, 1, 28)  # Middle dimension is the number of sample in one tensor.

X_train = tf.expand_dims(X_train, axis=2)
X_test = tf.expand_dims(X_test, axis=2)
print(tf.shape(X_train))

model = tf.keras.Sequential([
    InputLayer(input_shape=input_dim),
    FRMap(output_dim=24, filter=8),
    ReOrth(),
    ProjMap(),
    ProjPooling(pool_size=4),
    OrthMap(nb_eigen=20),
    ProjMap(),
    Flatten(),
    Dense(10)
])

print(model.summary())

model.compile(optimizer=Adam(1e-2), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=batch_size)

print(model.trainable_weights)
W = model.trainable_weights[0]

plt.imshow(W[0], cmap='gray')
plt.show()
