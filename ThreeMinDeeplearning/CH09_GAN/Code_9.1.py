"""
[GAN]

 - Generative Adversarial Network
 - Unsupervised learning -> Y를 사용하지 않음
 - 서로 대립(adversarial) 하는 두 신경망을 경쟁시켜가며 결과물을 생성하는 방법
 - 예를 들자면 위조지폐점(생성자, generator)가 경찰(구분자, discriminator)를 최대한 속이료 노력하고, 경찰은 위조한 지폐를 최대한 감별하려고 하는 것
 - 이 기법을 통하여 사진을 고흐 풍으로 다시 그려주거나, 선으로만 그려진 만화를 자동 채색한다는 등의 일이 가능

 - 이번 장에서는 GAN 모델을 이용하여 MNIST 손글씨 숫자를 무작위로 생성하는 예제 작성
"""

import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

total_epoch = 100
batch_size = 100
learning_rate = 0.0002
n_hidden = 256
n_input = 28*28
n_noise = 128 # 생성자의 입력값으로 사용할 노이즈의 크기, 랜덤한 노이즈를 입력하고 그 노이즈에서 손글씨 이미지를 무작위로 생성

# 모델
# Y는 없다
# 가짜 이미지는 노이즈에서 생성하므로 노이즈를 입력할 플레이스 홀더 Z 추가
X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성자 신경망 변수 설정
# 첫 번째 가중치와 편향 => 은닉층
# 두 번째 가중치와 편향 => 출력층
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden])) # tf.zeros() : 모든 값이 0인 텐서 생성
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 구분자 신경망 변수 설정
# 첫 번째 가중치와 편향 => 은닉층
# 두 번째 가중치와 편향 => 진짜와 얼마나 가까운지 판단, 0~1사이의 하나의 스칼라 값을 출력
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))

# 실제 이미지를 판별하는 구분자 신경망과 생성한 이미지를 판별하는 구분자 신경망은 같은 변수를 사용해야 함
# 그래야 진짜 이미지와 가짜 이미지의 특징을 동시에 잡아 낼 수 있기 때문

# 생성자 신경망 생성
# 무작위로 생성한 노이즈를 받아 가중치와 편향을 반영하여 은닉층 생성
# 은닉층에서 실제 이미지와 같은 크기의 결과값을 출력
def generator(noise_z):
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)

    return output

# 구분자 신경망 생성
# 같은 구성에 0~1사이 스칼라값 출력
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)

    return output

# 무작위 노이즈 생성 유틸 함수
def get_noise(batch_size, n_noise):
    return np.random.normal(size=(batch_size, n_noise))

# 노이즈 Z를 이용해 가짜 이미지를 만들 생성자 G 만들고, 이 G가 만든 가짜이미지와 진짜 이미지 X를 구문자에 넣어
# 입력한 이미지가 진짜인지 판별
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# 손실값
# 두가지가 필요
# 1. 생성자가 만든 이미지를 구분자가 가짜라고 판단하도록 하는 손실값(경찰 학습용)
# 2. 생성자가 만든 이미지가 진짜라고 판단하도록 하는 하는 손실값(위조지폐범 학습용)
# 경찰 학습하려면 D_real이 1에 가까워야 하고,
# 위조지폐범을 학습시키려면 D_gene를 0에 가까워야 함

# D_real과 1에서 D_gene를 뺀 값을 더한 값을 손실값으로 하여 이 값을 최대화하면 경찰학습이 이루어짐
loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# 위조지폐범 학습은 가짜 이미지 판별값을 D_gene를 1에 가깝게 만들기만 하면 됨
# 즉, 가짜 이미지를 넣어도 진짜 같다고 판별해야 함
# D_gene를 그대로 넣어 이를 손실값으로 하고, 이를 최대화 하면 위조지폐범 학습 가능
loss_G = tf.reduce_mean(tf.log(D_gene))

# 학습
# 주의점 : loss_D를 구할 때는 구분자 신경망에 사용되는 변수들만 사용하고, loss_G를 구할 때는 생성자 신경망에 사용되는 변수들만 사용하여 최적화
#         그래야 loss_D를 학습할 때 생성자가 변하지 않고, loss_G를 학습할 때 구분자가 변하지 않기 때문
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# loss를 최대화 해야 함
# minimize에 사용하는 loss_D와 loss_G에 음수부호 붙여 사용
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

# 학습
# 두개의 손실값을 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0 # loss_D와 loss_G의 결과값을 받을 변수

# 미니배치로 학습 반복
# 구분자는 X값을, 생성자는 노이즈인 Z값을 받으므로, 노이즈를 생성해주는 get_noise함수를 통해 배치 크기만큼 노이즈를 만들고 이를 입력
# 구분자와 생성자의 신경망을 각각 학습
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print('Epoch : ', '%04d' % epoch, 'D loss : {:.4}'.format(loss_val_D), 'G loss : {:.4}'.format(loss_val_G))

    # 학습 결과 확인
    # 10번째마다 생성기로 이미지를 생성하여 확인
    if epoch == 0 or (epoch + 1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        samples = sess.run(G, feed_dict={Z: noise})

        # 결과값을 28x28크기의 가짜 이미지로 만들어 samples 폴더에 저장
        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료!')



