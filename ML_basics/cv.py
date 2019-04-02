import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)

mnist= tf.keras.datasets.fashion_mnist
(ti,tl),(tri,trl) = mnist.load_data()

#normalizing
ti = ti/255.0
tri = tri/255.0

plt.imshow(ti[1])
print(ti[1])
print(tl[1])



#model
model= tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                           tf.keras.layers.Dense(1024,activation= tf.nn.relu),
                           tf.keras.layers.Dense(10,activation = tf.nn.softmax)])
model.compile(optimizer = tf.train.AdamOptimizer(),
               loss ='sparse_categorical_crossentropy',
               metrics= ['accuracy'])
model.fit(ti,tl,epochs =5)
test_loss = model.evaluate(tri,trl)

a = model.predict(ti)
print(a[0])
















