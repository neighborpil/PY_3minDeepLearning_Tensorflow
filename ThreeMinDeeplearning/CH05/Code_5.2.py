"""
[5.2 텐서보드 사용하기]
# 딥러닝은 대부분 학습시간이 상당히 오래 걸림
  효과적인 실험을 위하여 학습과정을 추적하는 것이 중요
  텐서플로는 텐서보드를 통하여 지원
# 텐서보드
  학습하는 중간중간 손실값이나 정확도 또는 결과물로 나온 이미지나 사운드 파일을 다양한 방식으로 시각화하여 보여줌
"""

import tensorflow.compat.v1 as tf
import numpy as np

# 데이터 및 플레이스홀더 준비
data = np.loadtxt('./data.csv', delimiter=',',
                  unpack=True, dtype='float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])

global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망의 각 계층에 다음 코드 추가
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

# 손실값 추적을 위하여 수집한 값을 지정
# tf.summary모듈의 scalar함수는 값이 하나인 텐서를 수집할 때 사용
tf.summary.scalar('cost', cost)

# 모델을 불러오거나 초기화
sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model2')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

# tf.summary.merge_all함수로 앞서 지정한 텐서들을 수집한 다음
# tf.summary.FileWriter함수를 이용해 그래프와 텐서들의 값을 저장할 디렉토리를 설정
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

# 최적화 실행
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # merged로 모아둔 텐서의 값들을 계산하여 수집
    # add_summary함수로 해당 값들을 앞서 지정한 디렉토리에 저장
    # 나중에 확인할 수 있도록 global_step값을 이용해 수집한 시점을 기록
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

# 모델을 저장
saver.save(sess, './model2/dnn.ckpt', global_step=global_step)

# 예측
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

"""
학습이 끝난 뒤 터미널을 열어서
tensorboard --logdir=./logs 를 입력

브라우저에 들어가서
localhost:6006으로 접속

"""