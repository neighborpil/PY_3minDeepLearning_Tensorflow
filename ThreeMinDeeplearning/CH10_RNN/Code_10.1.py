"""
[RNN]
 - RNN: Recurrent Nerual Network, 순환 신경망
 - 자연어 처리나 음성 인식처럼 순서가 있는 데이터를 처리
 - 앞의 정보로 다음에 나올 저보를 추측
 - 구글 신경망 기반 기계어 번역
 
 - RNN의 기본적 사용법
 - Sequence to Sequence 모델을 이용한 간단한 번역 프로그램

[10.1 MNIST를 RNN으로 학습하는 모델]
 - Cell : 한 덩어리의 신경망
 - 셀을 여러개 중첩하여 심층 신경망 생성
 - 앞 단계에서 학습한 결과를 다음 단계에서 학습
"""

import tensorflow.compat.v1 as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

learning_rate = 0.001
total_epoch = 30
batch_size = 128

n_input = 28 # 가로 픽셀 수
n_step = 28 # 세로 픽셀 수
n_hidden = 128
n_class = 10

# 기존과 다른 점은 n_input차원을 추가
# RNN은 순서가 있는 데이터를 다루므로 한 번에 입력받을 개수와 총 몇 단계로 이루어진 데이터를 받을지 설정해야 함
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# BasicRNNCell : 데이터 학습 시 맨 뒤에서는 맨 앞의 정보를 잘 기억 못함
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# LSTM(Long Short-Term Memory) : 이를 보완하기 위한 구조

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# outputs : [batch_size, n_step, n_hidden]
# -> [n_step, batch_size, n_hidden]으로 변경
outputs = tf.transpose(outputs, [1, 0, 2])
# -> [batch_size, n_hidden]
outputs = outputs[-1] # 마지막 차원 제거

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)

test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도 : ', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))