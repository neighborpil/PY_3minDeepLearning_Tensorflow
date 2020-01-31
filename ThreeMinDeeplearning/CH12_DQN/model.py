import tensorflow as tf
import numpy as np
import random
from collections import deque
class DQN:
    REPLAY_MEMORY = 10000 # 학습에 사용할 플레이 결과를 얼마나 많이 저장해서 사용할 지
    BATCH_SIZE = 32 # 한번 학습시 몇개의 기억을 사용할 지
    GAMMA = 0.99 # 오래된 상태의 가중치를 줄이기 위한 하이퍼 파라미터
    STATE_LEN = 4 # 한 번에 볼 프레임의 총 수

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque() # 게임 플레이 결과를 저장할 메모리
        self.state = None

        # 플레이스 홀더
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN]) # 게임의 상태[ None, 게임판의 가로, 게임판의 세로, 게임 상태의 개수(현재+과거+과거..)]
        self.input_A = tf.placeholder(tf.int64, [None]) # 각 상태를 만들어 낸 액션값, 원핫인코딩X, 행동 숫자 그대로 사용
        self.input_Y = tf.placeholder(tf.float32, [None]) # 손실값 계산에 사용할 값, 보상 + 목표 신경망으로 구한 다음 상태의 Q값임, 여기서 학습 신경망에서 구한 Q갑슬 뺀값을 손실값으로 학습 진행, Q값은 행동에 따른 가치 값, 목표 신경망에서 구한 Q값은 구한 값의 최대값(최적의 행동)을 학습 신경망에서 구한 Q값으로 사용

        # 학습을 진행할 신경망과 목표 신경망 구성
        # 두 신경망은 구성이 같으므로 같은 함수 사용 but 이름만 다르게
        self.Q = self._build_network('main')
        self.cost, self.train_op = self._build_op()

        self.target_Q = self._build_network('target')

    # 학습 신경망과 목표 신경망을 구성하는 함수
    # 상태값 input_X를 받아 행동의 가짓수만큼의 출력갑을 만듬
    # 이 값의 최대값을 취해 다음 행동을 정함
    def _build_network(self, name):
        with tf.variable_scope(name):
            # 풀링 계층이 없음, 이미지의 세세한 부분까지 판단하도록 하기 위하여
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding='same', activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding='same', activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)
    
            Q = tf.layers.dense(model, self.n_action, activation=None)
    
        return Q
    
    # DQN의 손실 함수 구한느 부분
    # 현재 상태를 이용하여 학습 신경망으로 구한 Q_value와 다음 상태를 이용하여 목표 신경망으로 구한 Q_value(input_Y)를 이용하여 손실값을 구하고 최적화 함
    def _build_op(self):
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1) # tf.multiply(self.Q, one_hot)함수는 self.Q로 구한 값에서 현재 행동의 인덱스에 해당하는 값만 선택하기 위해 사용
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    # 목표 신경망을 갱신하는 함수
    # 학습 신경망 변수들의 값을 목표 신경망으로 복사하여 목표 신경망의 변수들을 최신값으로 갱신
    def update_target_network(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    # 현재 상태를 이용해 다음에 취해야 할 행동을 찾는 함수
    # _build_network함수에서 계산한 Q_value를 이용
    def get_action(self):
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    # 학습
    # _sample_memory 함수를 이용하여 게임 플레이를 저장한 메모리에서 배치 크기만큼 샘플링
    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()
        
        # 메모리에서 다음 상태를 만들어 목표 신경망에 넣어 target_Q_value를 구함
        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        # 손실 함수에 보상값 입력, 게임 종료시 바로 넣고, 게임 진행중이면 보상값에 target_Q_value최대값을 추가하여 넣음, 현재 상태에서의 최대 가치를 목표로 삼기 위함
        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X: state,
                                                   self.input_A: action,
                                                   self.input_Y: Y})

    # 현재 상태 초기화
    # DQN에서 입력값으로 사용할 상태는 게임판의 현재 상태 + 앞의 상태를 몇개 합친것
    # 이를 입력값으로 만들기 우해 STATE_LEN만큼 스택으로 만듬
    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2) # axis가 2인 이유는 input_X를 넣을 플레이스 홀더가 [None, width, height, self.STATE_LEN]이기 때문

    # 게임 플레이 결과를 받아 메모리에 기억
    # 가장 오래된 상태 제거, 새로운 상태 만듬
    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    # 기억해둔 게임 플레이에서 임의의 메모리를 배치 크기만큼 가져옴
    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action , reward, terminal









