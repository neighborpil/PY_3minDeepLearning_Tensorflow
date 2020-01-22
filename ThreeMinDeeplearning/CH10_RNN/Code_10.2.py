"""
[10.2 단어 자동 완성]
 - 영문자 4개로 구성된 단어를 학습시켜 3글자만 주어지면 나머지 한 글자를 추천하여 단어를 완성
 - dynamic_rnn의 sequence_length 옵셔을 사용하여 가변 길이 단어를 학습
 - 여기서는 고정길이 단어 사용
"""

# 각각의 글자에 해당하는 인덱스를 원-핫 인코딩으로 표현한 값을  사용
import tensorflow.compat.v1 as tf
import numpy as np
char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z']

# {'a': 0', 'b': 1, 'c': 2, ...}
num_dic = {n: i for i, n in enumerate(char_arr)} # enumerate() : 인덱스와 값을 리턴
dic_len = len(num_dic)

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        input = [num_dic[n] for n in seq[:-1]]
        target = num_dic[seq[-1]]
        input_batch.append(np.eye(dic_len)[input]) # 입력값을 원-핫 인코딩으로 변경
        target_batch.append(target)

    return input_batch, target_batch

# 신경망 모델
learning_rate = 0.01
n_hidden = 128
total_epoch = 30

n_step = 3 # 처음 3글자
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# 심층 순환 신경망 생성
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})

    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))

print('최적화 완료')

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

input_batch, target_batch = make_batch(seq_data)

predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})

# 예측 단어 출력
predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n==예측 결과==')
print('입력값: ', [w[:3] + ' ' for w in seq_data])
print('예측값: ', predict_words)
print('정확도: ', accuracy_val)



