# All algorithims are here

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import open3d as o3d
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
        centroids = self.rng.choice(points, size=k, replace=False)

        assignment = np.zeros(len(points), dtype=int)

        assignment_prev = None
 
        iterations = 0
        start_time = time.perf_counter()
        while assignment_prev is None or any(assignment_prev!=assignment):
            assignment_prev= np.copy(assignment) 

            # First iterate over all points
            for i, point in enumerate(points):
                distances2 = np.linalg.norm(centroids - point, axis=1)
                closest_index = np.argmin(distances2)
                assignment[i] = closest_index

            # Second calculate new centroids
            for i in range(k):
                centroids[i] = points[assignment==i].mean(axis=0)

            iterations += 1
            current_time = time.perf_counter()
            elapsed_time = current_time - start_time
            minutes = elapsed_time / 60
            print(f"On Iteration {iterations}, {elapsed_time:.1f} seconds have elapsed ({minutes:.2F} minutes)")

        return centroids, assignment
    
    def pca(self):
        pass

    def svm(self):
        pass

if __name__ == "__main__":
    pass