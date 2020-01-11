"""
# 선형 회귀
 - 주어진 x와 y의 값으로 서로의 관계를 파악하는 것
 - 텐서플로우의 최적화 함수를 이용하여 x와 y의 상관관계를 분석하는 기초적인 선형 회기 모델 예시

"""

import tensorflow.compat.v1 as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# W(weight) : 가중치
# b(bias) : 편향
# x와 y의 상관관계를 설명하기 위한 변수인 W와 b를 각각 -1.0부터 1.0 사이의 균등분포(uniform distribution)를 가진 무작위 값으로 초기화
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# 자료를 입력받을 플레이스 홀더
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# X와 Y의 상관관계(여기서는 선형관계)를 분석하기 위한 수식을 작성
hypothesis = W * X + b

# 손실값 : 실제값과 모델로 예측한 값이 얼마나 차이가 나는가를 나타내는 값
# 손실 함수(loss function) : 한쌍(x, y)의 데이터에 대한 손실값을 계산하는 함수
# 손실값이 작을 수록 그 모델이 X와 Y의 관계를 잘 설명하고 있다는 뜻
# 비용(cost) : 손실을 전체 데이터에 대하여 구한 것
# 학습 : 변수들의 값을 다양하게 넣어 계산해보면서 이 손실값을 최소화 하는 W와 b의 값을 구하는 것
# 손실값으로는 예측값과 실제값의 거리를 가장 많이 사용
# 따라서 손실값은 예측값에서 실제값을 뺀 뒤 제곱
# 비용은 모든 데이터에 대한 손실값의 평균
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 함수 : 가중치와 편향값을 변경해가면서 손실값을 최소화하는 가장 최적화된 가중치와 편향값을 찾아주는 함수
# 이때 값들을 무작위로 변경하면, 시간이 너무 오래걸리고, 학습 시간 예측이 불가
# 따라서 빠르게 최적화 하기 위한 다양한 방법 사용, 경사하강법은 최적화 방법 중 가장 기본적인 알고리즘
# 경사하강법(gradient descent) 최적화 함수를 이용하여 손실값을 최소화하는 연산 그래프 생성
#  - 함수의 기울기를 구하고, 기울기가 낮은 쪽으로 계속 이동시키면서 최적의 값을 찾아나가는 방법
# 학습률(learning_rate) : 학습을 얼마나 급하게 할 것이냐를 설정하는 값
#  - 값이 너무 크면 손실갑을 찾지 못하고 지나치게 됨, 값이 너무 작으면 학습 속도가 느림
# 하이퍼파라미터(hyperparameter) : learning_rate와 같이 학습을 진행하는 과정에 영향을 주는 변수
#  - 머신러닝에서는 이 하이퍼파라미터를 잘 튜닝하는 것이 과제
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train_op = optimizer.minimize(cost)

# 파이선의 with 기능을 이용하여 세션 블록을 만들고, 세션 종료시 자원의 반환을 자동으로 처리(c#의 using과 같다)
# 변수들을 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 수행하는 그래프인 train_op를 실행
    # 실행시마다 변화하는 손실값을 출력
    # 학습은 100번 수행, feed_dict 매개변수를 통해, 상관관계를 알아내고자 하는 데이터인 x_data와 y_data를 입력해줌
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    # 학습으로 생성된 hypothesis를 가지고, y값을 예측
    print("X: 5, y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, y:", sess.run(hypothesis, feed_dict={X: 2.5}))



















