"""
[심층 신경망 구현]
# 딥러닝 : 신경망의 층을 둘 이상으로 구성한 것
# 다층신경망 만들기 : 신경망 모델에 가중치와 편향을 추가
"""
import tensorflow.compat.v1 as tf
import numpy as np

# [털, 날개]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

etc = [1, 0, 0]
mammal = [0, 1, 0]
bird = [0, 0, 1]

y_data = np.array([
    etc,
    mammal,
    bird,
    etc,
    etc,
    bird])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 다층 신경망 제작은 앞서 만든 신경망 모델에 가중치(weight)와 편향(bias)를 추가하기만 하면 됨
# 은닉층의 뉴런 수는 하이퍼파라미터이니 실험을 통해 가장 적절한 수를 정하면 됨
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.)) # [2, 10] -> [특징, 은닉층의 뉴런 수]
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.)) # [10, 3] -> [은닉층의 뉴런 수, 분류 수]

b1 = tf.Variable(tf.zeros([10])) # [10] -> [은닉층의 뉴런 수]
b2 = tf.Variable(tf.zeros([3])) # [3] -> [분류 수]

# 신경망 구성
# 특정 입력값에 대한 첫 번째 가중치와 편향, 그리고 활성 함수를 적용
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 출력층을 만들기 위해 두 번째 가중치와 편향을 적용하여 최종 모델을 만듬
model = tf.add(tf.matmul(L1, W2), b2)

# 앞의 기본 신경망 모델에서는 출력층에 활성화 함수를 적용하였으나, 사실 출력층에 보통 활성화 함수 사용 안함
# 은닉층과 출력응에서 활성화 함수를 적용할지 말지, 또 어떤 활성화 함수를 적용할지를 정하는 일은 신경망 모델을 만드는데 있어 가장 중요한 경험적, 실험적 요소임

# 손실함수 : 교차 엔트로피를 사용하나, 텐서플로우가 기본적으로 제공하는 교차 엔트로피 함수를 이용
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 최적화 함수 : AdamOptimizer를 사용, GradientDescentOptimizer보다 보편적으로 성능이 더 좋음
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step+1) % 10 == 0:
        print(step+1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print("예측값 : ", sess.run(prediction, feed_dict={X: x_data}))
print("실제값 : ", sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))