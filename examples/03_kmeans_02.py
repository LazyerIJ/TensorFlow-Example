import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn import KMeansClustering
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

class DataSet():

    def __init__(self):
        
        def randrange(mu,sig,num=200):
            return np.random.normal(mu,sig,num)

        a = np.array([randrange(2,2),randrange(3,2)]).T
        b = np.array([randrange(-2,1),randrange(-3,2)]).T
        c = np.array([randrange(-2,1),randrange(3,2)]).T
        d = np.array([randrange(2,2),randrange(-3,2)]).T
        self.data = np.concatenate([a,b,c,d])

    def getData(self):
        return self.data


class Config():

    def __init__(self,num_steps=2000,centroids=3):
        self.num_steps=num_steps
        self.centroids = centroids


if __name__=='__main__':

    myConfig = Config() 
    myDataset = DataSet()

    def input_fn(data):
        features = tf.constant(data,tf.float32,data.shape)
        labels = None
        return features,labels

    model = KMeansClustering(num_clusters=myConfig.centroids,
                             relative_tolerance=0.00001)

    model.fit(input_fn=lambda:input_fn(myDataset.data),
              steps=myConfig.num_steps)

    assignments = list(model.predict_cluster_idx(input_fn=lambda:input_fn(myDataset.data)))

    cmap = mpl.cm.Dark2.colors
    colors=[cmap[i] for i in assignments]
    plt.scatter(myDataset.data[:,0],myDataset.data[:,1],color=colors)
    plt.show()
    
    
