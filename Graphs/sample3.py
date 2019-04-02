import tensorflow as tf

with tf.name_scope("Scopea"):
    a= tf.add(1,2,name="a_add")
    b= tf.multiply(a,3,name="a_multiplytiply")
    
with tf.name_scope("scopeB"):
    c=tf.add(4,5,name="b_add")
    d=tf.multiply(c,6,name="b_multiplytiply")
    
e = tf.add(b, d, name="output")

writer = tf.summary.FileWriter('./graph3',graph=tf.get_default_graph())
writer.close()

# Example 2
graph = tf.Graph()

with graph.as_default():
    in_1 = tf.placeholder(tf.float32, shape=[], name="input_a")
    in_2 = tf.placeholder(tf.float32, shape=[], name="input_b")
    const = tf.constant(3, dtype=tf.float32, name="static_value")

    with tf.name_scope("Transformation"):

        with tf.name_scope("A"):
            A_multiply = tf.multiply(in_1, const)
            A_out = tf.subtract(A_multiply, in_1)

        with tf.name_scope("B"):
            B_multiply = tf.multiply(in_2, const)
            B_out = tf.subtract(B_multiply, in_2)

        with tf.name_scope("C"):
            C_div = tf.div(A_out, B_out)
            C_out = tf.add(C_div, const)

        with tf.name_scope("D"):
            D_div = tf.div(B_out, A_out)
            D_out = tf.add(D_div, const)

    out = tf.maximum(C_out, D_out)   

writer = tf.summary.FileWriter('./name_scope_2', graph=graph)
writer.close()


# To start TensorBoard after running this file, execute the following command:

# For Example 1
# $ tensorboard --logdir='./name_scope_1'

# For Example 2
# $ tensorboard --logdir='./name_scope_2'