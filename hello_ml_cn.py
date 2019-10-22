import tensorflow as tf
import numpy as np
import sys

NUM_STEPS = 500
MINIBATCH_SIZE = 100
TRAIN_CURRENT_BATCH = -1

""" 读入数据，这些标准TENSORFLOW 2.0数据形状不适合直接的计算，所以要进行格式转化
    x_train， x_test是二维28x28的图像信息矩阵，我把它转化成784的一维矢量
    y_train,  y_test是图像对应的数字，我把它转化成[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]矢量，数字对应的位置设置为比重1
"""
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

""" 将颜色变为 （0 - 1）之间进行单位比重计算 """
x_train, x_test = x_train / 255.0, x_test / 255.0

""" 每次取100个图片，把二维图片变为一维，重新组成矩阵 x, 矩阵形状【100， 784】
    对应的100个答案，组成矩阵 b，形状为【100，10】
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

""" 从测试组中取100个图片，转化成同样矩阵形状用于测试 W
"""
def test_batch():
    x_temp = x_test[100:200]
    x_ret = []
    for sub in x_temp:
        x_ret.append(sub.flatten())
    y_temp = y_test[100:200]
    y_ret = []

    for sub in y_temp:
        arr = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        arr[sub] = 1.
        y_ret.append(arr)
    return (x_ret, y_ret)

""" 根据 W x= b 建模 """

""" x 形状为 100 x 784 的矩阵  """
x = tf.compat.v1.placeholder(tf.float32, [None, 784])

""" 比重参数的模型 W 是784 x 10 的矩阵，这个矩阵就是机器学习找出来的，可用于人工智能预测计算等 """
W = tf.Variable(tf.zeros([784, 10]))

""" b 矩阵，就是结果 """ 
y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])

""" W 和 x 相乘，得出的是模型预测出来的矩阵 """ 
y_pred = tf.matmul(x, W)

""" 模型与真实结果的偏差 """ 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

""" 通过优化器改变 W 数值，使得结果更加接近 """ 
gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

""" y 矢量 （b）的10个值里最大值就是模型猜才出来的结果 """ 
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:
    # Train
    sess.run(tf.global_variables_initializer())

    """ 输入50000个矢量，让系统寻找 W """ 
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = train_next_batch()
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
        # Test

    """ 看看 W 的预测效果如何 """   
    test_xs, test_ys = test_batch()    
    ans = sess.run(accuracy, feed_dict={x: test_xs, y_true: test_ys})
    print ("Accuracy: {:.4}%".format(ans*100))
    
    """ 大概看看 W 的尊荣 """   
    np.savetxt('WMatrix.txt', W.eval(), fmt='"%.4f"')