"""
[DQN]
 - Deep Q-Learning
 - 강화학습으로 유명한 Q-Learning을 딥러닝으로 구현했다는 의미
 - 과거의 상태를 기억한 뒤 그 중에서 임의의 상태를 뽑아 학습
 - 손실값 계산을 위해 학습을 진행하면서 최적의 행동을 얻어내는 기본 신경망과
   얻어낸 값이 좋은 선택지인지 비교하는 목표 신경망을 분리하는 방법 사용
 - 목표 신경망은 일정 주기마다 갱신
 - 화면의 상태만으로 학습하므로, 이미지 인식에 뛰어난 CNN 이용

# Agent 구성
"""

import tensorflow as tf
import numpy as np
import random
import time

from game import Game
from model import DQN

# agent는 학습모드(train)와 게임 실행모드(replay)로 나뉜다
# 학습모드 때는 게임을 화면에 보여주지 않은채 빠르게 실행하여 학습 속도를 높임
# 실행모드에서는 학습된 결과를 이용하여 게임을 진행하면서 화면에 출력
# 이를 위하여 에이전트 실행 시 모드를 나누어 실행 할 수 있도록 tf.app.flags를 이용해 실행 시 받을 옵션 설정
tf.app.flags.DEFINE_boolean("train", False, "학습모드 - 게임 화면을 보여주지 않습니다")
FLAGS = tf.app.flags.FLAGS

MAX_EPISODE = 10000 # 최대 학습 게임 횟수
TARGET_UPDATE_INTERVAL = 1000 # 학습을 일정 횟수만큼 진행 할 때마다 한번씩 목표 신경망을 갱신하라는 옵션
TRAIN_INTERVAL = 4 # 게임 4프레임 마다 한번씩 학습
OBSERVE = 100 # 일정 수준의 학습 데이터가 쌓이기 전에는 학습하지 않고 지켜보기, 100프레임 이후부터 학습 진행

# 취할수 있는 행동 좌, 우, 상태유지
NUM_ACTION = 3 # 좌: 0, 유지: 1, 우: 2
SCREEN_WIDTH = 6
SCREEN_HEIGHT = 10

# 학습부
def train():
    print("뇌 세표 깨우는 중")
    sess = tf.Session()

    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=False)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    # 학습 결과 저장 및 학습 상태 확인
    rewards = tf.placeholder(tf.float32, [None]) # 한판(에피소드) 마다 얻는 점수를 저장하고 확인하기 위한 텐서
    tf.summary.scalar('avg.reward/ep.', tf.reduce_mean(rewards))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('logs', sess.graph)
    summary_merged = tf.summary.merge_all()

    # 목표 신경망 초기화
    brain.update_target_network()

    epsilon = 1.0 # 무작위 행동 선택값, 게임 진행중에는 줄어듬
    time_step = 0 # 학습 진행 조절 위한 진행된 프레임(상태) 횟수
    total_reward_list = [] # 학습 결과 확인 위하여 점수 저장할 리스트

    for episode in range(MAX_EPISODE):
        terminal = False # 게임 종료 상태
        total_reward = 0 # 한 게임당 얻은 총 점수

        state = game.reset() # 게임의 상태 초기화
        brain.init_state(state) # 상태를 DQN에 초기 상태값으로 넣어줌, 상태는 screen_witdh * screen_height 크기의 화면 구성

        # 원래 DQN에서는 픽셀값들을 상태값으로 받지만, 여기에서 사용하는 Game 모듈에서는 학습 속도를 높이고자 해당 위치에 사각형이 있는지 없는지 1과 0으로 전달

        while not terminal:
            if np.random.rand() < epsilon:
                action = random.randrange(NUM_ACTION)
            else:
                action = brain.get_action()

            if episode > OBSERVE:
                epsilon -= 1/ 1000

            # 게임 진행
            state, reward, terminal = game.step(action)
            total_reward += reward

            # 현재 상태를 신경망 객체에 기억
            brain.remember(state, action, reward, terminal)

            # 4프레임마다 학습 진행
            if time_step > OBSERVE and time_step % TRAIN_INTERVAL == 0:
                brain.train()

            # 1000 프레임 마다 목표신경망 갱신
            if time_step % TARGET_UPDATE_INTERVAL == 0:
                brain.update_target_network()

            time_step += 1

        # 사각형애 충돌해 게임이 종료되면 획득한 점수 출력 및 저장
        print('게임횟수: %d 점수: %d' % (episode+1, total_reward))

        total_reward_list.append(total_reward)

        # 게임 10번마다 로그 저장
        if episode % 10 == 0:
            summary = sess.run(summary_merged, feed_dict={rewards: total_reward_list})
            writer.add_summary(summary, time_step)
            total_reward_list = []

        # 게임 100번마다 학습된 모델 저장
        if episode % 100 == 0:
            saver.save(sess, 'model/dqn.ckpt', global_step=time_step)

# 실행 함수
def replay():
    print("뇌 세표 깨우는 중")
    sess = tf.Session()

    # 트레이닝 중에는 False
    game = Game(SCREEN_WIDTH, SCREEN_HEIGHT, show_game=True)
    brain = DQN(sess, SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ACTION)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    for episode in range(MAX_EPISODE):
        terminal = False
        total_reward = 0

        state = game.reset()
        brain.init_state(state)

        while not terminal:
            action = brain.get_action()
            state, reward, terminal = game.step(action)
            total_reward += reward

            brain.remember(state, action, reward, terminal)

            time.sleep(0.3)
        
        print('게임횟수: %d 점수: %d' % (episode+1, total_reward))

# 학습용으로 실행할지, 결과로 실행할지 선택
def main(_):
    if FLAGS.train:
        train()
    else:
        replay()

if __name__ == '__main__':
    tf.app.run()

