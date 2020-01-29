"""
    This is a file you will have to fill in.
    It contains helper functions required by K-means method via iterative improvement
"""
import numpy as np
from random import sample

def init_centroids(k, inputs):
    """
    Selects k random rows from inputs and returns them as the chosen centroids
    :param k: number of cluster centroids
    :param inputs: a 2D Numpy array, each row of which is one input
    :return: a Numpy array of k cluster centroids, one per row
    """
    k_cluster_centroids = np.array(sample(inputs.tolist(), k))
    return(k_cluster_centroids)



def assign_step(inputs, centroids):
    """
    Determines a centroid index for every row of the inputs using Euclidean Distance
    :param inputs: inputs of data, a 2D Numpy array
    :param centroids: a Numpy array of k current centroids
    :return: a Numpy array of centroid indices, one for each row of the inputs
    """

    centroid_indices = np.zeros(len(inputs))
    #print(centroids)
    for i in range(len(inputs)):
        cent_comp_list = []
        for c in range(len(centroids)):
            entry = np.linalg.norm(inputs[i]-centroids[c])**2
            #print(entry)
            cent_comp_list.append(entry)
        #print(len(cent_comp_list))
        centroid_indices[i] = cent_comp_list.index(min(cent_comp_list))
    #print(np.array(centroid_indices))
    return(np.array(centroid_indices))



def update_step(inputs, indices, k):
    """
    Computes the centroid for each cluster
    :param inputs: inputs of data, a 2D Numpy array
    :param indices: a Numpy array of centroid indices, one for each row of the inputs
    :param k: number of cluster centroids, an int
    :return: a Numpy array of k cluster centroids, one per row
    """
    summing_centroids = np.zeros((k,np.size(inputs,1)))
    updated_centroids = np.zeros((k,np.size(inputs,1)))
    #print(indices)
    #print(np.shape(indices))
    count = np.zeros(k)
    for i in range(len(inputs)):
        label = int(indices[i])
        #print(label)
        summing_centroids[label] = summing_centroids[label] + inputs[i]
        count[label] = count[label] + 1
    for p in range(len(count)):
        updated_centroids[p] = summing_centroids[p] / count[p]
    return(updated_centroids)


def kmeans(inputs, k, max_iter, tol):
    """
    Runs the K-means algorithm on n rows of inputs using k clusters via iterative improvement
    :param inputs: inputs of data, a 2D Numpy array
    :param k: number of cluster centroids, an int
    :param max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int
    :param tol: relative tolerance with regards to inertia to declare convergence, a float number
    :return: a Numpy array of k cluster centroids, one per row
    """
    level = 100000
    count = 0
    initial_centroids = init_centroids(k, inputs)

    while count < max_iter or tollevel > tol:
        centroid_indices = assign_step(inputs, initial_centroids)
        updated_centroids = update_step(inputs, centroid_indices, k)

        tollevel = 100 * ( (np.sum(np.absolute(updated_centroids - initial_centroids)) / np.sum(np.absolute(initial_centroids))) )
        count = count + 1

        initial_centroids = updated_centroids

    return(updated_centroids)
