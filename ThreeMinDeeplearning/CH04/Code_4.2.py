"""
[간단한 분류 모델 만들기]
# 딥러닝이 가장 폭넓게 활용되는 분야는 패턴 인식을 통한 영상처리 분야

# 분류(classification) : 패턴을 파악하여 여러 종류로 구분하는 작업

# 털과 날개가 있느냐를 기준으로 포유류와 조류를 구분하는 신경망 모델 제작

"""

import tensorflow.compat.v1 as tf
import numpy as np

# 1. 변수설정
# 학습에 사용할 데이터 정의 [털, 날개]의 이진 데이터, 있으면 1, 없으면 0
# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# 각 개체가 실제 어떤 종류인지를 나타내는 레이블(분류)값
# 원-핫 인코딩(one-hot encoding) : 데이터가 가질 수 있는 값들을 일렬로 나열한 배열을 만들고, 그 중 표현하려는 값을 뜻하는
#  인덱스의 원소만 1로 표기하고, 나머지 원소는 모두 0으로 표시
# 레이블 데이터는 원-핫 인코딩으로 표시
# 판별하고자 하는 개체의 종류는 기타=0, 포유류=1, 조류=2의 세가지[기타, 포유류, 조류]
etc = [1, 0, 0]
mammal = [0, 1, 0]
bird = [0, 0, 1]

#y_data = np.array([
#    etc,
#    mammal,
#    bird,
#    etc,
#    etc,
#    bird])

y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]])

# 2. 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치와 편향 변수 설정
# W : [입력층(특징 수), 출력층(레이블 수)], -1.0 ~ 1.0사이로 랜덤하게 설정
# b : 레이블 수인 3개의 요소를 가진 변수로 설정
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

# 이 가중치를 곱하고 편향을 더한 결과를 활성화 함수(ReLU)에 적용하면 신경망 구성은 끝
# tf.matmul() : 행렬 곱
L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L)

# 출력값을 softmax함수를 이용하여 사용하기 쉽게 다듬아 줌
# softmax 함수 : 배열 내의 결과값들의 전체 합이 1이 되도록 만들어 줌
model = tf.nn.softmax(L)

# 손실 함수는 원-핫 인코딩을 이용한 대부분의 모델에서 사용하는 교차 엔트로피(Cross-Entropy)함수를 사용
# 교차 엔트로피(Cross-Entropy) : 예측값과 실제 값 사이의 확률분포 차이를 계산한 값

# Y : 실측값, model : 신경망을 통해 나온 예측값
# Y * tf.log(model) : model값에 log를 취한 값을 Y와 곱함
# tf.reduce_sum(Y * tf.log(model), axis=1) : 행별로 값을 다 더함
# tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1)) : 배열 안 값의 평균을 구함
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# reduce_XXX 함수들으 텐서의 차원을 줄여줌, XXX 부분이 구체적인 차원 축소 방법
# axis 매개변수로 축소할 차원을 정함
# 예를들어 reduce_sum(<입력 텐서>, axis=1)은 주어진 텐서의 1번째 차원의 값들을 다 더해(값 1개로 만들어) 그 차원을 없앤다는 뜻
# sum외에 prod, min, max, mean all, any, logsumexp등을 제공

# 3. 학습
# 기본적인 경사하강법으로 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션을 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 앞서 구성한 특징과 레이블 데이터를 이용하여 학습을 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 학습 도중 10번에 한 번씩 손실값을 출력
    if(step + 1) % 10 == 0:
        print(step+1, sess.run(cost,feed_dict={X: x_data, Y: y_data}))

# 학습된 결과를 확인
# tf.argmax(input, axis=?) : input에서 가장 큰 값의 인덱스를 반환한다
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

# 정확도 출력
# 전체 학습 데이터에 대한 예측값과 실측값을 tf.equal함수로 비교한 뒤
is_correct = tf.equal(prediction, target)
# true/false값으로 나온 결과를 다시 tf.cast함수를 이용하여 0과 1로 바꾸어 평균을 냄
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100 , feed_dict={X: x_data, Y: y_data}))



