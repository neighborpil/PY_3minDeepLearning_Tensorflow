import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


hello = tf.constant('Helo, Tensorfow!'); # tensor라는 자료형이고, 상수를 담고 있음
print(hello)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
#print(c)

"""
 tensro : 텐서플로에서 다양한 수학식을 계산하기 위한 가장 기본적이고 중요한 자료형
 랭크 : 차우너의 수
 세이프 : 차원의 요소 개수, 텐서의 구조를 설명해줌

 그래프 : 텐서들의 연산 모임
 텐서 플로 : 텐서와 텐서의 연산들을 먼저 정의하여 그래프를 만듬, 이후 필요할 때
            연산을 실행하는 코드를 넣어 원하는 시점에 실제 연산을 수행(지연실행)
"""

sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()