"""
[6.1 MNIST 학습하기]

# MNIST
  손으로 쓴 숫자들의 이미지를 모아놓은 데이터 셋
  0~9까지의 숫자를 28x28픽셀크기의 이미지로 구성
"""

import tensorflow.compat.v1 as tf

# MNIST데이터를 내려 받고 원-핫 인코딩 방식으로 읽어들임
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot=True)

# 한번에 여러개의 데이터를 학습시키면 효율이 좋지만 그만큼 많은 메모리와 컴퓨팅 성능 요구
# 따라서 데이터를 적당한 크기로 잘라서 학습(미니배치, minibatch)
# 텐서의 첫번째 차원은 None으로 지정
# 이는 한번에 학습시킬 MNIST 이미지의 개수를 지정하는 값이 들어감
# None으로 지정하면 텐서플로우가 알아서 계산
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 모델 제작
# 2개의 은닉층이 다음처럼 구성된 신경망 제작
# 786(입력, 특징 개수) -> 256(첫 번째 은닉층 뉴런 개수) ->256(두 번째 은닉층 뉴런 개수) -> 10(결과값 0~9 분류)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))
model = tf.add(tf.matmul(L2, W3), b3)

# 손실값 계산 및 최적화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits= model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 초기화 및 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 미니배치 설정
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size) # 미니 배치의 개수

# MNIST 데이터 전체를 학습하는 일을 총 15번 반복
# 에포크(epoch) : 학습 데이터 전체를 한 바퀴 도는 것을 에포크
for epoch in range(15):
    total_cost = 0

    # 한 에포크 내에서 미니배치 사이즈의 총 개수만큼 반복
    for i in range(total_batch):
        # 학습할 데이터를 배치 사이즈만큼 가져오고 입력값(batch_xs)과 출력값(batch_ys)에 저장
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 최적화시키고 손실값을 가져와 저장
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    # 한 세데의 학습이 끝나면 학습한 세대의 평균 손실값을 출력
    print('Epoch : ', '%04d' % (epoch+1), 'Avg. cost = ', '{:.3f}'.format(total_cost/total_batch))

print('최적화 완료')

# 결과 예측
# 예측 결과인 model의 값과 실제 레이블인 Y의 값을 비교
# 예측값 : tf.argmax(model, 1)
# 실제값 : tf.argmax(Y, 1)
# tf.argmax()는 두번째 차원(1)의 가장 높은 값을 가지는 인덱스를 반환
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
# tf.cast() : is_correct를 0과 1로 변환, 변환한 값들의 평균(tf.reduce_mean)을 구함
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print('정확도: ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))