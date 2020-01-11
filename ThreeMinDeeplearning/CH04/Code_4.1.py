"""
[CH04 기본 신경망 구현]

# 인공 신경망(artificial neural network) : 입력 신호(X)에 가중치(W)를 곱하고, 편향(b)을 더한 뒤
  활성화 함수(Sigmoid, ReLU 등)를 거쳐 결과값 y를 만들어 내는 것
# 학습(learning) 또는 훈련(training) : 원하는 y값을 얻기 우하여 W와 b의 값을 변경해가면서 적절한 값을 찾아내는 최적화 과정
# 활성화 함수(activation function) : 인공 신경망을 통과해온 값을 최종적으로 어떤 값으로 만들지 결정
  - Sigmoid, ReLU, tanh등이 있음
# Sigmoid : 0 - 1.0사이
# ReLU : 0보다 작으면 항상 0을, 0보다 크면 입력값을 그대로 출력
# tanh : -1.0 ~ 1.0

# 제한된 볼트만 머신(Restricted Boltzmann Machine, RBM)
# 드롭아웃 기법, ReLU

# 역전파(backpropagation) : 출력층이 내 놓은 결과의 오차를 신경망을 따라 입력층까지 역으로 전파하며 계산해나가는 방식
 - 입력층부터 가중치를 조절해 가는 기존의 방식보다 훨씬 유의미한 방식으로 가중치를 조절해 주어 최적화 과정이 훨씬 빠르고 정확
"""

