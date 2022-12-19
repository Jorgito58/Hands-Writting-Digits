# Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data(path="mnist.npz")
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = keras.models.Sequential([
                                 keras.layers.Conv2D(28,kernel_size=(3,3)),
                                 keras.layers.MaxPooling2D(),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(128,activation='relu'),
                                 keras.layers.Dense(10,activation = 'softmax')
])

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(x_train,y_train,epochs=10)

predictions = model.predict([x_test])
print(np.argmax(predictions[100]))
plt.imshow(x_test[100], cmap='gray')
# Save the entire model as a SavedModel.
model.save('D:\\Software Development\\PythonLearning\\Curso-Python\\0. Python_Proyects\\HandsWritting_Numbers\\saved_model\\handsw_conv_model')
