import preprocess
import model
import numpy as np
import sys
import random
import tensorflow as tf


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def run(params):

    files = ["./person01.aac", './person02.aac', './person03.aac']

    save_path = "./trained_model.ckpt"

    data, num_examples = preprocess.feature_extract(files)

    output_size = len(files)

    name = "test"


    training_epochs = 500
    batch_size = 100

    sess = tf.Session()
    dnn = model.Model(sess=sess, output_size=output_size, name="test")

    sess.run(tf.global_variables_initializer())

    print('Learning Started!')

    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, data['X'], data['Y'])

            c, _ = dnn.train(batch_xs, batch_ys)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost)



    print('Learning Finished!')

    # 테스트
    data_test, test_num_examples = preprocess.feature_extract(['./test01.wav'], 0, 3)
    print("test: ", data_test["X"].shape, data_test["Y"].shape)
    print("Prediction: ", data_test['X'])
    print("Accuracy: ", dnn.get_accuracy(x_test=data_test['X'], y_test=data_test['Y']))

    # 예측값과 실제값
    r = random.randint(0, test_num_examples - 1)
    print("Label: ", sess.run(tf.argmax(data_test['Y'][r:r + 1], 1)))
    print(dnn.predict(x_test=data_test['X'][r:r + 1]))
    print("Prediction: ", sess.run(tf.argmax(dnn.predict(x_test=data_test['X'][r:r + 1]), 1)))

    # 학습 모델 저장
    saver = tf.train.Saver
    saver.save(sess, save_path=save_path)

