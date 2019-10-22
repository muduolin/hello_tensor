import tensorflow as tf
import numpy as np
import sys

NUM_STEPS = 500
MINIBATCH_SIZE = 100
TRAIN_CURRENT_BATCH = -1

""" Read Tensorflow MNIST test datasets
    x_train， x_test are arrays of 28x28 2-d images, I need to convert them into arrays of vectors (size 784) 
    y_train,  y_test are arrays of result，I need to convert them into array of [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] vector for matrix multiplication, the position with result would be set to weight 1
"""
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

""" convert the color into range (0, 1) for easier interpretation  """
x_train, x_test = x_train / 255.0, x_test / 255.0

""" 
    The function will fetch a batch of 100 images, format them into an array of image vectors, this is the x in the equation Ax = b, the matrix shape is [100, 784]
    Also format the answers into vectors, this is the b in the equation Ax = b, the matrix shape is [100,10]
"""
def train_next_batch():
    global TRAIN_CURRENT_BATCH
    TRAIN_CURRENT_BATCH = TRAIN_CURRENT_BATCH + 1
    x_temp = x_train[TRAIN_CURRENT_BATCH*MINIBATCH_SIZE:(TRAIN_CURRENT_BATCH+1)*MINIBATCH_SIZE]
    x_ret = []
    for sub in x_temp:
        x_ret.append(sub.flatten())
    y_temp = y_train[TRAIN_CURRENT_BATCH*MINIBATCH_SIZE:(TRAIN_CURRENT_BATCH+1)*MINIBATCH_SIZE]
    y_ret = []

    for sub in y_temp:
        arr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        arr[sub] = 1.
        y_ret.append(arr)
    return (x_ret, y_ret)

""" Now, just take 1000 images from the test batch for testing, this is to measure the accuracy of matrix A once it is optimized by Tensorflow. The data is converted to the same shape as the traning set
"""
def test_batch():
    x_temp = x_test[1000:2000]
    x_ret = []
    for sub in x_temp:
        x_ret.append(sub.flatten())
    y_temp = y_test[1000:2000]
    y_ret = []

    for sub in y_temp:
        arr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        arr[sub] = 1.
        y_ret.append(arr)
    return (x_ret, y_ret)

""" Linear Model Ax = b """

""" x is a 100 x 784 matrix  """
x = tf.compat.v1.placeholder(tf.float32, [None, 784])

""" weight matrix is 784 x 10, tensorflow would be able to calculate and optimize this matrix, then we can use it for prediction """
W = tf.Variable(tf.zeros([784, 10]))

""" b matrix is the result """ 
y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])

""" Suppose we have the matrix A, Ax will give us the predcited result """ 
y_pred = tf.matmul(x, W)

""" now we find the differences between model and true value """ 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

""" adjust the A matrix to optimize the result """ 
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

""" the max value in y vector （b） is the predicting result """ 
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # start the training
    sess.run(tf.global_variables_initializer())

    """ I feed the engine with 50000 vectors, let the engine calculate the best matrix A """ 
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = train_next_batch()
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    """ test A matrix's accuraxy with data from test batch """   
    test_xs, test_ys = test_batch()    
    ans = sess.run(accuracy, feed_dict={x: test_xs, y_true: test_ys})
    print ("Accuracy: {:.4}%".format(ans*100))
    
    """ save the matrix to a file to view its glory """   
    np.savetxt('AMatrix.txt', W.eval(), fmt='"%.4f"')