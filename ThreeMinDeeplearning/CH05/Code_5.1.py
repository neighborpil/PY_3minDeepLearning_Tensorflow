"""
[CH5 : 텐서보드와 모델 재사용]
 - 텐서보드를 이용해 손실값의 변화를 그래프로 추적
[5.1 : 학습 모델 저장하고 재사용하기]

# data.csv 파일 로드
[털, 날개, 기타, 포유류, 조류]
"""

import tensorflow.compat.v1 as tf
import numpy as np

# np.loadtxt() : 데이터 읽어들임
# unpack=True : 결과의 행과 열을 전치(transpose) 시킨다. False가 기본값
#data = np.loadtxt('./data.csv', delimiter=',',
#                  unpack=True, dtype='float32')

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32') # 행렬을 전치시킨다
"""
print(data)

# unpack=False
[[0. 0. 1. 0. 0.]
 [1. 0. 0. 1. 0.]
 [1. 1. 0. 0. 1.]
 [0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 1. 0. 0. 1.]]

# unpack=True
[[0. 1. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 1.]
 [1. 0. 0. 1. 1. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 1.]]
"""

x_data = np.transpose(data[0:2]) # 전치된 행렬읠 0~1행까지를 가져와 다시 전치시킴으로써 분리
y_data = np.transpose(data[2:])

"""
print(x_data)
print(y_data)

[[0. 0.]
 [1. 0.]
 [1. 1.]
 [0. 0.]
 [0. 0.]
 [0. 1.]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
"""

# 신경망 모델 구성

# global_step : 신경망 모델을 저장할 때 쓸 변수
# 학습에 직접 사용되지는 않고, 학습 횟수를 카운트하는 변수
# 따라서 변수 정의시 trainable=False옵션 줌
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step') # initial_value = 0

# 앞 장보다 계층을 하나 더 늘리고, 편향은 없이 가중치만 사용한 모델 작성

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.)) # 신경망의 은닉층을 높이면 복잡도가 높은 문제 해결에 도움되나, 과적합 문제 발생
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost, global_step=global_step) # 최적화 함수가 변수들을 최적화 할 때마다 global_step변수의 값이 1씩 증가

# 세션 열고 최적화
# tf.global_variables() : 앞서 정의한 변수들을 가져오는 함수
# 이 함수를 이용하여 앞서 정의한 변수들을 모두 가져와, 이후 이 변수들을 파일에 저장하거나 이전에 학습한 결과를 불러와 담는 변수들로 사용
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

# ./model 디렉토리에 기존에 학습해 둔 모델이 있는지 확인
# saver.restore 함수를 사용하여 학습된 값을 불러오고
# 아니면 변수를 새로 초기화
# 체크포인트 파일(checkpoint file) : 학습된 모델을 저장한 파일
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("restored")
else:
    sess.run(tf.global_variables_initializer())
    print("newly created")

# 최적화 수행
for step in range(20):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step), 'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 학습한 변수들을 지정한 체크포인트 파일에 저장
# 두번째 매개변수 : 파일의 위치와 이름
# 세번째 매개변수 : global_step의 값은 저장할 파일의 이름에 추가적으로 붙게 됨, 텐서 변수 또는 숫자값을 넣어 줄 수 있음
#                  이를 통하여 여러 상태의 체크포인트를 만들 수 있고, 가장 효과적인 체크포인트를 선별해서 사용 가능
saver.save(sess, './model/dnn.ckpt', global_step=global_step)

# 예측 결과와 정확도를 확인
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

"""
# 이런 방식으로 학습시킨 모델을 저장하고 불러와서 재사용 가능
  이 방식을 응용하여 모델 구성, 학습, 예측 부분을 각기 분리하여 학습을 따로 한 뒤, 예측만 단독으로 실행하는 프로그램 작성 가능

"""