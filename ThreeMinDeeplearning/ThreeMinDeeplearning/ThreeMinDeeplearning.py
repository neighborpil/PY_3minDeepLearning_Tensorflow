import tensorflow as tf




"""
#Code
hello = tf.constant("Helo, Tensorflow!")
print(hello)

# Result
Tensor("Const:0", shape=(), dtype=string)
"""



"""
#Code
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)
sess.close();

# Result
Tensor("Add:0", shape=(), dtype=int32)
"""


"""
#Code
hello = tf.constant("Helo, Tensorflow!")
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)

sess = tf.Session()
print(sess.run(hello))
print(sess.run(c))
sess.close();

# Result
b'Helo, Tensorflow!'
42
"""



"""
#Code
X = tf.placeholder(tf.float32, [None, 3]) # float의 변수가 3개 있어야 한다
print(X)

x_data = [[1, 2, 3], [4, 5, 6]] # x_data : 2행 3열

W = tf.Variable(tf.random_normal([3, 2])) # W : 3행 2열인데 랜덤값으로 초기화
b = tf.Variable(tf.random_normal([2, 1])) # tf.random_normal()정규분포 무작위 값으로 초기화

expr = tf.matmul(X, W) + b # tf.matmul() : 행렬곱, 외적, 2행3열 X 3행2열 => 2행2열

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('x_data : {}'.format(x_data))
print('W : {}'.format(sess.run(W)))
print('b : ', sess.run(b))
print('expr : ', sess.run(expr, feed_dict={X: x_data}))

sess.close();

# Result
x_data : [[1, 2, 3], [4, 5, 6]]
W : [[-1.0318967   2.66031   ]
 [ 0.77901685 -0.58803695]
 [-1.2936598  -0.03109844]]
b :  [[1.7478626]
 [2.1285217]]
expr :  [[-1.6069801  3.1388035]
 [-5.86594    9.642986 ]]
"""

x_data = [1, 2, 3]
y_data = [1, 2, 3]
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# placeholders
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name='Y')


"""
# Title
# 선형회기모델 구현

#Code

# Result

"""


"""
#Code

# Result

"""

