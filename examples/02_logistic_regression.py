import tensorflow as tf
import numpy as np

from sklearn.datasets import make_moons,make_circles,make_blobs
from matplotlib import pyplot as plt


def make_polynom(x,y,dims):

    poly_result=np.transpose([x,y])

    for dim in range(dims):
        poly_result = np.c_[poly_result,np.power(x,dim)*np.power(y,dims-dim)]
    return poly_result

class Model():

    def __init__(self,learning_rate,input_dims):

        self.input_dims = input_dims

        self.X = tf.placeholder(tf.float32, [None,input_dims+2])
        self.Y = tf.placeholder(tf.float32, [None,2])

        self.W = tf.Variable(tf.random_normal([input_dims+2,2],stddev=0.01),name='w_p1')
        self.b = tf.Variable(tf.random_normal([1,2],stddev=0.01),name='b_p1')
        self.logits = tf.nn.sigmoid(tf.add(tf.matmul(self.X,self.W),self.b))

        self.cost = tf.losses.log_loss(self.Y, self.logits, scope="loss")
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.cost)

    def get_train(self,sess,x,y):
        poly_x = make_polynom(x[:,0], x[:,1], self.input_dims)
        feed_dict={self.X:poly_x, self.Y:y}
        loss,_ = sess.run([self.cost,self.train],feed_dict=feed_dict)
        return loss
    
    def get_pred(self,sess,x):
        poly_x = make_polynom(x[:,0], x[:,1], self.input_dims)
        feed_dict={self.X:poly_x}
        logit = sess.run(self.logits,feed_dict=feed_dict)
        pred = tf.argmax(tf.nn.softmax(logit),1).eval()
        return pred

if __name__=='__main__':

    learning_rate=0.01
    poly_dims=5
    n_epochs=30
    batch_size=50
    plot_idx=1
    n_samples=800
    colors=['b','y']

    train_samples = int(n_samples*0.5)

    #X,y = make_circles(n_samples=n_samples, factor=0.2, noise=0.01)
    X,y = make_moons(n_samples=n_samples,noise=0.001,random_state=0,shuffle=True)
    train_X = X[:train_samples]
    train_y = y[:train_samples]
    test_X  = X[train_samples:]
    test_y  = y[train_samples:]

    myModel = Model(0.01,poly_dims)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs+1): 

            total_loss=0.0
            shuffle_idx = np.random.permutation(train_samples)
            train_X = train_X[shuffle_idx]
            train_y = train_y[shuffle_idx]

            if epoch%int(n_epochs/3)==0:

                plt.subplot(150+plot_idx)
                plt.title('[epoch] {}'.format(epoch))

                pred = myModel.get_pred(sess,test_X)
                color = [colors[i] for i in pred]
                plt.scatter(test_X[:,0], test_X[:,1],color=color)
                plot_idx+=1
                print('[+]plot data at epoch {0:4d}'.format(epoch))

            for step in range(int(train_samples/batch_size)):

                s_idx = step*batch_size
                x_train = train_X[s_idx:s_idx+batch_size]
                y_train = train_y[s_idx:s_idx+batch_size]
                y_train = tf.one_hot(indices=y_train,depth=2).eval()

                total_loss += myModel.get_train(sess=sess,
                                                x=x_train,
                                                y=y_train)
    
            batch_loss = total_loss / batch_size
            print('[*][epoch]{0:3d}/{1} [cost]{2:.4f}'.format(epoch,n_epochs,batch_loss))
    plt.subplot(155)
    plt.title('[data]')
    color=[colors[i] for i in test_y]
    plt.scatter(test_X[:,0],test_X[:,1],color=color)

    plt.show()



