"""
    This is the class file you will have to fill in using helper functions defined in kmeans.py.
"""
import numpy as np

from kmeans import kmeans

class KmeansClassifier(object):
    """
    K-Means Classifier via Iterative Improvement
    @attrs:
        k: The number of clusters to form as well as the number of centroids to generate (default = 10), an int
        tol: Relative tolerance with regards to inertia to declare convergence, the default value is set to 0.0001, a float number
        max_iter: the maximum number of times the algorithm can iterate trying to optimize the centroid values, an int,
                  the default value is set to 500 iterations
        cluster_centers_: a Numpy array where each element is one of the k cluster centers
    """

    def __init__(self, n_clusters = 10, max_iter = 500, threshold = 1e-4):
        """
        Initiate K-Means with some parameters
        """
        self.k = n_clusters
        self.tol = threshold
        self.max_iter = max_iter
        self.cluster_centers_ = np.array([])

    def train(self, X):
        """
        Compute K-Means clustering on each class label and store your result in self.cluster_centers_
        :param X: inputs of training data, a 2D Numpy array
        :return: None
        """
        self.cluster_centers_ = kmeans(X, self.k, self.max_iter, self.tol)

    def predict(self, X, centroid_assignments):
        """
        Predicts the label of each sample in X based on the assigned centroid_assignments.

        :param X: A dataset as a 2D Numpy array
        :param centroid_assignments: a Numpy array of 10 digits (0-9) representing the interpretations of the digits of the plotted centroids
        :return: A Numpy array of predicted labels
        """

        centroid_indices = np.zeros(len(X))
        #print(centroids)
        for i in range(len(X)):
            cent_comp_list = []
            for c in range(len(self.cluster_centers_)):
                entry = np.linalg.norm(X[i]-self.cluster_centers_[c])**2
                #print(entry)
                cent_comp_list.append(entry)
            #print(len(cent_comp_list))
            centroid_indices[i] = centroid_assignments[cent_comp_list.index(min(cent_comp_list))]
        #print(np.shape(np.array(centroid_indices)))
        return(np.array(centroid_indices))

    def accuracy(self, data, centroid_assignments):
        """
        Compute accuracy of the model when applied to data
        :param data: a namedtuple including inputs and labels
        :return: a float number indicating accuracy
        """
        pred = self.predict(data.inputs, centroid_assignments)
        return np.mean(pred == data.labels)
