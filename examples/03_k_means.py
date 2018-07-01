import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def choose_random_centroids(samples, n_clusters):
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0,n_samples))
    begin = [0,]
    size = [n_clusters,]
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroies = tf.gather(samples, centroid_indices)
    return initial_centroies

def create_samples(n_clusters, n_samples_per_cluster, n_features,
                   embiggen_factor, seed):
    np.random.seed(seed)
    slices=[]
    centroids=[]

    for i in range(n_clusters):

        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32,
                                   seed=seed, name="cluster_{}".format(i))

        current_centroid = (np.random.random((1,n_features)) * embiggen_factor)- (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)

    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0 , name='centroids')
    return centroids, samples

def plot_clusters(epoch,all_samples, centroids, n_samples_per_cluster):
    plt.title('{0}'.format(epoch))
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))

    for i, centroid in enumerate(centroids):
        samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:,0], samples[:,1], c=colour[i])
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k',
                 mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m',
                 mew=5)

def assign_to_nearest(samples, centroids):

    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    t_subtracted = tf.subtract(expanded_vectors, expanded_centroids)
    distances = tf.reduce_sum( tf.square(t_subtracted), 2)
    nearest_indices = tf.argmin(distances, 0)
    return nearest_indices
  
def update_centroids(samples, nearest_indices, n_clusters):
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

if __name__=='__main__':
    n_features = 2
    n_clusters = 3
    n_samples_per_cluster = 500
    seed = 700
    embiggen_factor = 70
    np.random.seed(seed)
    plot_idx=1
    training_iter=8

    centroids , samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
    initial_centroids = choose_random_centroids(samples,n_clusters)
    nearest_indices = assign_to_nearest(samples,initial_centroids)
    updated_centroids = update_centroids(samples,nearest_indices,n_clusters)

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        samples = samples.eval()

        for step in range(training_iter):

            _,centroids_values = sess.run([nearest_indices,updated_centroids])
            plt.subplot(100+training_iter*10+plot_idx)
            plot_clusters(step,samples, centroids_values, n_samples_per_cluster)
            plot_idx+=1

    plt.show()


