import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from layers import FRMap, ReOrth, ProjMap, ProjPooling, OrthMap
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputLayer, Dense, Flatten, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()

new_X = []
for x in X_train:
    q,r = np.linalg.qr(x/255)
    new_X.append(q)
X_train = np.stack(new_X)

new_X = []
for x in X_test:
    q,r = np.linalg.qr(x / 255)
    new_X.append(q)
X_test = np.stack(new_X)


plt.imshow(X_train[0], cmap="gray")
plt.show()

#X_train = X_train / 255
#X_test = X_test / 255

batch_size = 128
#           (d0, q)     first and last dimensions are data dimension: (d,q).
input_dim = (28, 28)  # Middle dimension is the number of sample in one tensor.

model = tf.keras.Sequential([
    InputLayer(input_shape=input_dim),
    FRMap(output_dim=26, filter=64),
    ReOrth(),
    ProjMap(),
    ProjPooling(),
    OrthMap(nb_eigen=20),
    ProjMap(),
    Flatten(),
    Dense(10)
])

print(model.summary())

model.compile(optimizer=Adam(5e-3), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=batch_size)

print(model.trainable_weights)
W = model.trainable_weights[0]

plt.imshow(W[0], cmap='gray')
plt.show()
