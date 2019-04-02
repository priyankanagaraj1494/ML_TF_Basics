import tensorflow as tf
def inference(X):
    return
def inputs():
    return
def loss(X,Y):
    return
def train(total_loss):
    return
def evaluate(sess,X,Y):
    return

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    X,Y = inputs()
    total_loss = loss(X,Y)
    train_op= train(total_loss)
    coord = tf.train.Coordinator() #to coordinate the termination of a set of threads.
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("epoch:",step,"loss: ", sess.run([total_loss]))
            #print("final model W", sess.run(W), "b:", sess.run(b))
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
