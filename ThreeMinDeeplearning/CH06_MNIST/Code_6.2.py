"""
[6.2 드롭아웃]

 - Overfitting : 학습데이터에 너무 잘 맞추어져서 그 외의 데이터와는 잘 안맞는 상황
 - Overfitting을 방지하기 위하여 Dropout 사용

# Dropout
 - 학습 시 전체 신경망 중 일부만을 사용하도록 하는 것
 - 학습 단계마다 일부 뉴런을 제거(사용하지 않도록) 함으로써, 일부특징이 특정 뉴런들에 고정되는 것을 막아
   가중치의 균형을 잡도록 하는 것
 - 학습시 시간이 조금 더 걸림
"""

import tensorflow.compat.v1 as tf

# 데이터셋
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 모델
# 드랍아웃 기법을 사용해 학습하더라도, 학습이 끝난 뒤 예측시에는 신경망 전체를 사용하도록 해줘야 함
# 따라서 keep_prob라는 플레이스홀더를 만들어 학습시에는 0.8, 예측시에는 1을 넣어 신경망 전체 사용하도록 해야 함
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))
model = tf.add(tf.matmul(L2, W3), b3)

# 손실값 계산 및 최적화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 미니 배치 설정
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# 학습
for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob:0.8})
        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch+1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

print("Train finished")

# 예측
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도 : ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
