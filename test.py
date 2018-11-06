import tensorflow as tf
import model

def run(params):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    save_path = "./trained_model.ckpt"

    saver = tf.train.Saver
    saver.restore(sess, save_path)


    test_model = model.Model(sess=sess, name="result", output_size=params[2])



