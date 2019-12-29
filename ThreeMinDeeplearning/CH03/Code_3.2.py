import tensorflow as tf
"""
# 플레이스 홀더 : 그래프에 사용할 입력값을 받기 위하여 사용되는 매개변수
변수 : 그래프를 최적화하는 용도로 텐서플로우가 학습한 결과를 갱신하기 위하여 사용하는 변수


"""

# None은 크기가 정해지지 않았음을 의미
X = tf.placeholder(tf.float32, [None, 3]) # 두번째 차원은 요소를 3개씩 가지고 있음.  ex)[[1, 2, 3], [4, 5, 6]]
print(X)

x_data = [[1, 2, 3], [4, 5, 6]]

# W와 b에 텐서플로우의 변수를 생성하여 할당
W = tf.Variable(tf.random_normal([3, 2])) # [3, 2] 행렬 형태의 텐서
b = tf.Variable(tf.random_normal([2, 1])) # [2, 1] 행렬 형태의 텐서
# => tf.random_normal함수를 이용하여 정규분포의 무작위 값으로 초기화

# 입력값과 변수들을 계산할 수식 작성
# X와 W가 행렬이기 때문에 tf.matmul함수를 사용
# 행렬이 아닐 경우에는 단순히 곱셈 연산자(*)나 tf.mul함수를 사용하면 됨
expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer()) # tf.global_variables_initializer() : 앞서 정의한 변수들을 초기화 하는 함수

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict = {X: x_data})) # feed_dict 매개변수 : 그래프를 실행할 때 사용할 입력값을 지정

sess.close()

