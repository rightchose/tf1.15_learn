import tensorflow as tf
tf.enable_eager_execution()

""" y = x^2 """
x = tf.get_variable('x', shape=[1], initializer=tf.constant_initializer(3.))

with tf.GradientTape() as tape: # 在tf.GradientTape()的上下文内，所有计算步骤都会被记录用于求导
    y = tf.square(x)

y_grad = tape.gradient(y,x)
print([y.numpy(), y_grad.numpy()])

X = tf.constant([[1.,2.], [3.,4.]]) # [2,2]
y = tf.constant([[1.], [2.]]) # [2,1]

w = tf.get_variable('w', shape=[2,1], initializer=tf.constant_initializer([[1.],[2.]]))
b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer([1.]))
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X,w) + b - y))

w_grad, b_grad = tape.gradient(L, [w,b])
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])

X = tf.constant(X)
y = tf.constant(y)

a = tf.get_variable('a', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
b = tf.get_variable('b', dtype=tf.float32, shape=[], initializer=tf.zeros_initializer)
variables = [a, b]

num_epoch = 100
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for e in range(num_epoch):
    print("epoch {}".format(e+1))
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
