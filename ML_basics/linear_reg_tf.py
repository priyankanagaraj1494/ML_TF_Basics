import tensorflow as tf
import os
import utils
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_file = "./birth_life_2010.txt"
batch_size = 10

def read_birth_life_data(filename):
    """
    Read in birth_life_2010.txt and return:
    data in the form of NumPy array
    n_samples: number of samples
    """
    text = open(filename, 'r').readlines()[1:]
    data = [line[:-1].split('\t') for line in text]
    births = [float(line[1]) for line in data]
    lifes = [float(line[2]) for line in data]
    data = list(zip(births, lifes))
    n_samples = len(data)
    data = np.asarray(data, dtype=np.float32)
    return data, n_samples

data, n_samples = read_birth_life_data(data_file)

#tf.data.Dataset.from_tensor_slices((X,Y))
train_data = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
#train_data = train_data.batch(batch_size)
#test_data = tf.data.Dataset.from_tensor_slices((data[:,0],data[:,1]))
#test_data = test_data.batch(batch_size)
print(train_data.output_types)
print(train_data.output_shapes)
#creating placeholders for x,y
#X = tf.placeholder(dtype = tf.float32 ,shape =None, name = "X")
#Y = tf.placeholder(dtype = tf.float32 ,shape =None, name = "Y")

epochs = 50

#iterator = tf.data.Iterator.from_structure(train_data.output_types,train_data.output_shapes)
iterator = train_data.make_initializable_iterator()
X,Y =iterator.get_next()
w = tf.get_variable("weights",shape= None, dtype = tf.float32,initializer = tf.zeros([2,1]))
b = tf.get_variable("bias",shape= None, dtype = tf.float32,initializer = tf.constant(0))
#train_init = iterator.make_initializer(train_data)
#test_init = iterator.make_initializer(test_data)"""



def inference(X):
    return tf.matmul(X,w)+ b

def loss():
    return tf.square(Y - inference(X), name = "loss")

def optimizer(loss):
    opt = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)
    return opt

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./graph", sess.graph)

    for i in range(epochs):
        sess.run(iterator.initializer)
        total_loss= 0
        _,loss = sess.run([optimizer,loss], feed_dict={X: x,Y: y})
        total_loss+= loss
        print("loss:",total_loss)
    w_out,b_out = sess.run([w,b])
    print("w:",w_out)
    print("b:",b_out)
    writer.close()