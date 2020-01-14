"""
# 지도학습(supervised learning)
 - 프로그램에게 원하는 결과를 알려주고 학습하는 방법
 - X와 Y 모두 있는 상태에서 학습

# 비지도학습(unsupervised learning)
 - 입력값으로부터 데이터의 특징을 찾아내는 학습 방법
 - X만 있는 상태에서 학습
 - 오토인코더 : 가장 널리 쓰이는 비지도 학습 신경망

[오토 인코더(Autoencoder)]

 - 입력값과 출력값을 같게 하는 신경망
 - 가운데 계층의 노드 수가 입력값보다 적은 것이 특징
 - 이를 통해 데이터를 압축하는 효과를 얻게 되고, 노이즈 제거에 효과적

 - 입력층으로 들어온 데이터를 인코더를 통해 은닉층으로 내보내고, 은닉층 데이터를 디코더를 통해 출력층으로 내보냄
   그 뒤 출력값과 입력값이 비슷해지도록 만드는 가중치를 찾는 것
"""

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

# 하이퍼 파라미터
learning_rate = 0.01 # 최적화 함수에 사용할 학습률
training_epoch = 20 # 전체 데이터를 학습할 총 횟수
batch_size = 100 # 미니배치로 한번에 학습할 사이즈
n_hidden = 256 # 은닉층의 뉴런 갯수
n_input = 28*28 # 입력값의 크기

# 모델
X = tf.placeholder(tf.float32, [None, n_input])
# 비 지도 학습이므로,Y값이 없음

# 인코더
# 가중치와 편향 변수를 원하는 뉴런의 개수만큼 설정하고
# 그 변수들을 입력값과 곱하고 더한 뒤, 활성화 함수인 sigmoid 함수를 적용
# 중요한 것은 n_input값보다 n_hidden의 값이 더 작아야 함
# 이를 통해 입력값을 압축하고 노이즈를 제거하면서 입력값의 특징을 찾게됨
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))

encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# 디코더
# 입력값은 은닉층의 크기로, 출력값을 입력층의 크기로 만듬
W_decoder = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_deocder = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decoder), b_deocder))

# 위한 손실 함수
# 오토인코더의 목적은 출력값을 입력값과 가장 비슷하게 만드는 것
# 이렇게 하면 압축된 은닉층의 뉴런들을 통해 입력값의 특징을 알아낼 수 있다
# 디코더가 내보낸 결과값과 실측값(X)의 차를 손실값으로 설정
cost = tf.reduce_mean(tf.pow(X - decoder, 2))

# 가중치를 최적화
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})

        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch+1), 'Avg. cost = ', '{:.4f}'.format(total_cost / total_batch))

print('최적화 완료')

# 결과 확인
# 정확도가 아닌, 디코더로 생성해낸 결과를 직관적인 방법으로 확인
# matplotlib를 이용 이미지로 출력
# 총 10개의 테스트 데이터를 가져와 디코더를 이용해 출력값으로 만듬
sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

# numpy모듈을 이용해 MNIST 데이터를 28x28 크기의 이미지 데이터로 재구성
# matplotlib의 imshow함수를 이용해 그래프에 이미지로 출력
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()
