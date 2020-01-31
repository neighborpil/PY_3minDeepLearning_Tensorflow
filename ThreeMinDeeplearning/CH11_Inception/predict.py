"""
[이미지 예측 스크립트]

# pb파일
 - 학습 결과를 protocol buffer라는 데이터 형시그로 저장해둔 파일

# 주의사항
 - tensorflow 2.0에서는 compat.v1해도 오류가 나며 실행이 안된다
   2.0을 지우고 1.4로 다운그레이드 하여 실행가능, 깃에 이슈가 있는거 보니 아마 버그가 있는듯 하다
"""

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

tf.app.flags.DEFINE_string("output_graph", "./workspace/flowers_graph.pb", "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels", "./workspace/flowers_labels.txt", "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image", True, "이미지 추론 후 이미지를 보여줍니다.")
FLAGS = tf.app.flags.FLAGS

def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]

    # pb파일을 읽어들여 신경망 그래프를 생성
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # 읽어들인 신경망 모델에서 예측에 사용할 텐서 지정, 저장되어 있는 모델에서 최종 출력층은 final_result:0라는 텐서
        logits = sess.graph.get_tensor_by_name('final_result:0')
        # 예측 스크립트를 실행할 때 주어진 이름의 이미지 파일을 읽어들인 뒤, 그 이미지를 예측 모델에 넣어 예측 실행
        # DecodeJpeg/contents:0은 플레이스 홀더
        image = tf.gfile.FastGFile(sys.argv[1], 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

    print('===예측 결과===')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%' % (name, score * 100))

    if FLAGS.show_image:
        img = mpimg.imread(sys.argv[1])
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    tf.app.run()


