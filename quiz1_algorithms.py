# All algorithims are here

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

class Algorithms:

    def __init__(self):
        self.rng = np.random.default_rng(4)

    def k_means(self, points, k):
        #_points (array): data points in the shape of [number of data points, number of features]
        # k (int): number of clusters
        print(f"Attempting to Cluster {len(points)} points into {k} clusters")
        # TODO initialise the centroids using rng.choice() function within the domain of the point set
        # IMPORTANT: Two centroids cannot be the same number
        centroids = self.rng.choice(points, size=k, replace=False)

        # Assign points to clusters

        # TODO: initialize an array with zeros equivalent to number of points. Each point will have an assignment [0/1/2].
        assignment = np.zeros(len(points), dtype=int)

        assignment_prev = None
        # TODO write the condition for iteration
        # Tip 1. Starting the assignment
        # Tip 2. When to stop the iteration
        iterations = 0
        start_time = time.perf_counter()
        while assignment_prev is None or any(assignment_prev!=assignment): #ans : assignment_prev is None or any(assignment_prev != assignment)
            assignment_prev= np.copy(assignment) # First keep track of the latest assignment

            # First iterate over all points
            for i, point in enumerate(points):

                # TODO calculate the Euclidean distance (distance2) from each point to the centroids
                # Tip: distance2 should be of length 3
                distances2 = np.linalg.norm(centroids - point, axis=1)
                # TODO calculate the closest centroid for the point
                # Tip: see how np.argmin(...) works
                closest_index = np.argmin(distances2)
                # TODO populate the assignment array with the corresponding centroid
                assignment[i] = closest_index

            # Second calculate new centroids
            for i in range(k):
                # TODO replace centroids with the mean of points assigned to that centroid
                # Tip: Only select the points assigned to each index value
                centroids[i] = points[assignment==i].mean(axis=0)

            iterations += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            minutes = elapsed_time / 60
            print(f"On Iteration {iterations}, {elapsed_time:.1f} seconds have elapsed ({minutes:.2F} minutes)")

        return centroids, assignment
    
    def kmeans_ignore_ground(self, points, k, ground_threshold=0.2):

        # Detect ground
        ground_mask = points[:, 2] < ground_threshold

        ground_points = points[ground_mask]
        non_ground_points = points[~ground_mask]

        # Run kmeans only on objects
        centroids, obj_assignment = self.k_means(non_ground_points, k)

        # Combine assignments
        assignment = np.zeros(len(points), dtype=int)

        assignment[ground_mask] = 0
        assignment[~ground_mask] = obj_assignment + 1

        # Add ground centroid (optional)
        if len(ground_points) > 0:
            ground_centroid = np.mean(ground_points, axis=0)
            centroids = np.vstack([ground_centroid, centroids])

        return centroids, assignment

    def pca(self):
        pass

    def svm(self):
        pass

if __name__ == "__main__":
    pass