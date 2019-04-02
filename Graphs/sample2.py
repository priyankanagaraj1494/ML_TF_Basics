import tensorflow as tf

a = tf.constant(5,name="inputa")
b = tf.constant(4,name = "inputb")
c = tf.multiply(a,b,name ="mul")
d = tf.add(a,b,name = "add")
e = tf.add(c,d,name= "final")

sess = tf.Session()
sess.run(e)
writer = tf.summary.FileWriter('./graph',sess.graph)
writer.close()
sess.close()
