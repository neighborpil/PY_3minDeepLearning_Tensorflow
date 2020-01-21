"""
[9.2 원하는 숫자 생성하기]
 - 노이즈에 레이블 데이터를 힌트로 넣어주는 방버


"""
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28*28
n_noise = 128
n_class = 10

# 플레이스 홀더에 Y값 추가
# 결과값 판정용이 아닌, 노이즈와 실제 이미지에 각각 해당하는 숫자를 힌트로 넣어주는 용도
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성자 신경망
# 변수를 생성하지 않고, tf.layers를 사용
# GAN 모델은 생성자와 구분자를 동시에 학습, 따라서 학습 시 각각 신경망 변수를 따로 학습
# 하지만 tf.layers를 사용하면 변수를 선언하지 않고 다음과 같이 tf.variable_scope를 이용하여 스코프 지정 가능
# 이렇게 하면 나중에 이 스코프에 해당하는 변수들만 따로 불러 올 수 있음
def generator(noise, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise, labels], axis=1) # tf.concat() : axis에 해당하는 차원 결합, 0은 행, 1은 열이라 보면 쉽다
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input, activation = tf.nn.sigmoid)

    return output

# 생성자 신경망과 같은 방법으로 구분자 신경망 생성
# 주의점 : 구분자는 진짜 이미지를 판별할 때와 가짜 이미지를 판별할 때 똑같은 변수를 사용해야 함
# 그러기 위하여 scope.reuse_variables 함수를 이용해 이전에 사용한 변수를 재사용하도록 작성해야 함
# 출력값(output)에 활성화 함수를 사용하지 않는데, 앞서와 다르게 손실값 계산에 sigmoid_cross_entrophy_with_logits 함수를 사용하기 위함
def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)

    return output

# 노이즈 생성 유틸리티 함수, 노이즈를 균등분포로 생성하도록 작성
def get_noise(batch_size, n_noise):
    return np.random.uniform(-1., 1., size=[batch_size, n_noise])

# 생성자를 구성하고 진짜 이미지 데이터와 생서앚가 만든 이미지 데이터를 이용하는 구분자를 하나씩 생성
# 생성자에는 레이블 정보를 추가하여 추후 레이블 정보에 해당하는 이미지를 생성할 수 있도록 유도
# 구분자에서는 진짜 이미지 구분자에서 사용한 변수들을 재사용하도록 reuse옵션을 True로 설정
G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)

# 손실함수
# loss_D는 loss_D_real와 loss_D_gene를 합친 것, 이 값을 최소화 함으로써 구분자(경찰) 학습 가능
# 이렇게 하려면 D_real은 1에 가까워야 하고(실제 이미지는 진짜라고 판별), D_gene는 0에 가까워야 함(생성한 이미지는 가짜라고 판별)
# 이를 위해 loss_D_real은 D_real의 결과값과 크기만큼 1로 채운 값들을 비교(ones_like함수)
# loss_D_gene는 D_gene의 결과값과 D_gene의 크기만큼 0으로 채운 값들을 비교(zeros_like함수)
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))
loss_D = loss_D_real + loss_D_gene

# 그 뒤 loss_G를 구함
# loss_G는 생성자(위조지폐범)를 학습사키기 위한 손실값
# sigmoid_cross_entrophy_with_logits 함수를 이용하여 D_gene를 1에 가깝게 만드는 값을 손실값으로 취하도록 함
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))

# tf.get_collection 함수를 이용해 discriminator와 generator 스코프에서 사용된 변수를 가져옴
# 이 변수들을 최적화에 사용할 각각의 손실 함수와 함께 최적화 함수에 넣어 학습 모델 구성
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)

# 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Y: batch_ys, Z: noise})

    print('Epoch:', '%04d' % epoch, 
          'D loss: {:4}'.format(loss_val_D), 
          'G loss: {:4}'.format(loss_val_G))

    # 확인용 이미지 생성
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Y: mnist.test.labels[:sample_size], Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()

            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')


