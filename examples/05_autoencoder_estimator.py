import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import tensorflow.contrib.layers as lays
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


def resize_batch(imgs):

    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))

    return resized_imgs

def cnn_model_fn(features, labels, mode):

    net = lays.conv2d(features["x"], 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    logits = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)

    predictions = {"output":logits}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.reduce_mean(tf.square(logits - labels))  # claculate the mean square error loss
    logging_hook = tf.train.LoggingTensorHook({ "global_step":tf.train.get_global_step(),
                                               "loss":loss},
                                              every_n_iter=20)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=[logging_hook])

def main(unused_argv):

    training_num=10000
    test_num=7
    num_epochs=10
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    print('[*]resizing imgs...') 

    train_data = resize_batch(mnist.train.images[:training_num,:])
    eval_data = resize_batch(mnist.test.images[:test_num,:])

    print('[*]resizing finish!!')
    print('training_data : {}'.format(train_data.shape))
    print('eval_data     : {}'.format(eval_data.shape))
    print('\n\n')
          

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="mnist_Autoencoder_model")

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_data,
        batch_size=200,
        num_epochs=num_epochs,
        shuffle=False)

    mnist_classifier.train(
        input_fn=train_input_fn)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        num_epochs=1,
        shuffle=False)

    eval_results = list(mnist_classifier.predict(input_fn=predict_input_fn))

    idx=1
    for p in eval_results:
        plt.subplot(2,test_num,idx)
        plt.imshow(eval_data[idx-1][:,:,0],cmap='gray')
        plt.subplot(2,test_num,test_num+idx)
        plt.imshow(p["output"][:,:,0], cmap='gray')
        idx+=1
    plt.show()

if __name__ == "__main__":
    tf.app.run()
