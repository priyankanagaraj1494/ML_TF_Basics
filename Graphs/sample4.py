import tensorflow as tf
import numpy as np

graph= tf.Graph()

with graph.as_default():
    with tf.name_scope("variales"):
        global_step= tf.Variable(0,dtype=tf.int32, name="global_step")
        increment_step= global_step.assign_add(1) #increment by 1
        prev_value=tf.Variable(0.0,dtype=tf.float32,name="prev_val")
    with tf.name_scope("exercise"):
        with tf.name_scope("input"):
            a= tf.placeholder(tf.float32,shape=[None],name="place_a")
        with tf.name_scope("intermediate_layer"):
            b=tf.reduce_prod(a,name="prodb")
            c=tf.reduce_sum(a,name="sumc")
        with tf.name_scope("output"):
            d= tf.add(b,c,name="addd")
            output = tf.subtract(d,prev_value,name="output")
            update_prev= prev_value.assign(output)
    with tf.name_scope("summaries"):
        tf.summary.scalar("Output", output)  # Creates summary for output node
        tf.summary.scalar("product of inputs", b)
        tf.summary.scalar("sum of inputs", c)
    with tf.name_scope("global_ops"):
        init = tf.initialize_all_variables() #init global-var
        print(init)
        merged_summaries= tf.summary.merge_all()
        print(merged_summaries)

sess= tf.Session(graph= graph)
writer = tf.summary.FileWriter("./improved_graph",graph= graph)
sess.run(init)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    output, summary, step = sess.run([update_prev, merged_summaries, increment_step], feed_dict=feed_dict)
    print(output)
    print(summary)
    print(step)
    writer.add_summary(summary, global_step=step)

run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])
run_graph([2,8])
run_graph([3,1,3,3])
run_graph([8])
run_graph([1,2,3])
run_graph([11,4])


writer.flush()
writer.close()
sess.close()