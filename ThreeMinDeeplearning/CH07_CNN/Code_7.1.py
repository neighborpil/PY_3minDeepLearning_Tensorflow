"""
[CH7. 합성곱 신경망(Convolutional Neural Network, CNN)]

 - 이미지 인식분야에서 강력한 성능
 - 컨볼루션 계층(Convolution layer, 합성곱 계층)과 풀링 계층(pooling layer)로 구성
 - (2D 컨볼루션의 경우) 2차원 평면 행렬에서 지정한 영역의 값들을 하나의 값으로 압축하는 것

# 컨볼루션 계층
 - 하나의 값으로 압축시 컨볼루션 계층은 가중치와 편향을 적용
# 풀링 계층
 - 단순히 값들 중 하나를 선택해 가져옴

# 윈도우
 - 지정한 크기의 영역
 - 윈도우의 값을 오른쪽 및 아래쪽으로 한 칸씩 움직이며 은닉층을 완성
 - 움직이는 크기 또한 변경 가능
 - 몇 칸씩 움직일지 정하는 값을 스트라이드(stride)라고 함

 - 입력층의 윈도우를 은닉층의 뉴런 하나로 압축할 때, 컨볼루션 계층에서는 윈도우 크기만큼의 가중치와 1개의 편향이 필요
 - 예를들어 윈도우의 크기가 3x3일 경우 3x3의 가중치와 1개의 편향을 커널(kernel) 또는 필터(filter)라고 한다
 - 이 커널은 해당 은닉층을 만들기 위한 모든 윈도우에 공통적으로 적용

# CNN의 특징
 - 기본신경망으로 28x28의 입력층이 있을 경우, 모든 뉴런을 연결하면 784개의 가중치를 찾아야 함
 - 하지만 CNN에서는 3x3의 9개의 가중치만 찾으면 됨
 - 따라서 계산량이 적어져 학습이 더욱 빠르고 효율적임

 - 단점으로는 복잡한 특징을 가진 이미지들을 분석하기에는 부족
 - 보통은 커널을 여러개 사용

 - 커널의 크기 및 개수 역시 하이퍼파라미터의 하나로써 분석하고자 하는 내용에 따라 중요한 요소임
"""

import tensorflow.compat.v1 as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 앞장의 기본 신경망 모델에서는 28x28짜리 하나의 차원으로 구성
# CNN에서는 2차원 평면으로 구성 => 좀 더 직관적
# X의 첫번째 차원인 None은 입력 데이터의 개수
# 마지막 차원 1은 특징의 개수, MNIST 데이터는 gray scale이라 채널에 색상이 하나 뿐이므로 1을 사용
# 출력값 10개 분류
# keep_prob : 드롭아웃
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# CNN 계층 구성

# 컨볼루션 계층
# 3x3 사이즈의 커널을 가진 컨볼루션 계층 제작
# 텐서플로우가 제공하는 tf.nn.conv2d함수 사용
# X : 입력층, W1 : 가중치
# 오른쪽과 아래쪽으로 한칸씩 움직이는 32개의 커널을 가진 컨볼루션 계층을 만듬
# padding='SAME'옵션은 커널 슬라이딩시 이미지의 가장 외곽에서 한칸 밖으로 움직이는 옵션, 이렇게 함으로써 좀 더 정확하게 평가 가능
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)

# 풀링 계층
# 컨볼루션 계층을 입력층으로 사용
# 커널의 크기는 2x2
# strides=[1, 2, 2, 1]값은 슬라이딩 시 두 칸씩 움직이겠다는 옵션
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 두번째 계층
# 32는 첫 번째 컨볼루션 계층의 커널 개수 => 첫 번째 컨볼루션 계층의 출력 개수이며, 찾아낸 이미지의 특징 개수

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 추출한 특징들을 이용하여 10개의 분류를 만들어내는 계층 구성
# 직전 풀링 계층의 크기가 7x7x64이므로 tf.reshape함수를 이용해 7 * 7 * 64 크기의 일차원 계층 생성
# 이 배열 전체를 최종 출력값의 중간 단계인 256개의 뉴런으로 연결하는 신경망 생성
# 완전 연결 계층(fully connected layer) : 인접한 계층의 모든 뉴런과 상호 연결된 계층
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

# L3의 출력값 256개를 받아 최종 출력값인 0~9ㄹ이블을 갖는 10개의 출력값을 만듬
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

# 손실함수 및 최적화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# optimzer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

# 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch : ', '%04d' % (epoch+1), 'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : ', sess.run([accuracy], feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                                  Y: mnist.test.labels,
                                                  keep_prob: 1}))


