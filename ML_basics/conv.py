import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras



print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(ti,tl),(tri,trl) = mnist.load_data()
ti=ti.reshape(60000, 28, 28, 1)
ti=ti/255.0
tri = tri.reshape(10000,28,28,1)
tri = tri/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128,activation = 'relu'),
                                    tf.keras.layers.Dense(10,activation='softmax')])#softmax classification
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(ti,tl,epochs=5)
model.summary()
test_loss =model.evaluate(tri,trl)

#visualizing conv and pooling
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(ti[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(ti[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(ti[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
