# All algorithims are here

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.linalg import svd
import open3d as o3d
import numpy as np
import time

from sklearn.preprocessing import StandardScaler

class Algorithms:

    def __init__(self):
        self.rng = np.random.default_rng(4)

    def k_means(self, points, k):
        """
        Run KMeans using scikit-learn for stability and speed.
        """
        print(f"Clustering {len(points)} points into {k} clusters using sklearn KMeans...")
        scaled = StandardScaler()
        scaled = scaled.fit_transform(points)
        kmeans = KMeans(n_clusters=k)
        
        kmeans.fit(scaled)
        
        centroids = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
        print(f"Found {len(np.unique(cluster_labels))} unique clusters.")
        
        return centroids, cluster_labels
    
    def pca(self, X, n_components=3):
        """
        Perform PCA using scikit-learn.

        Parameters:
            X : np.ndarray of shape (n_points, 3)
            n_components : int, number of principal components to keep

        Returns:
            pc_vectors : np.ndarray (n_components, 3) principal directions
            explained_variance : np.ndarray (n_components,) variance along each component
        """
        pca_model = PCA(n_components=n_components)
        pca_model.fit(X)

        pc_vectors = pca_model.components_        # shape (3,3)
        explained_variance = pca_model.explained_variance_

        return pc_vectors, explained_variance

    def extract_cluster_features(self, points: np.ndarray, cluster_labels: np.ndarray, k: int):
        
        pca_centers = np.zeros((k, 3))
        pca_pc1 = np.zeros((k, 3))
        pca_pc2 = np.zeros((k, 3))
        pca_pc3 = np.zeros((k, 3))
        features = np.zeros((k, 3))  # linearity, planarity, scattering
        eigvals = np.zeros((k, 3))
        for i in range(k):
            cluster_points = points[cluster_labels == i]

            if len(cluster_points) == 0:
                # empty cluster, skip
                continue

            # Cluster center
            center = cluster_points.mean(axis=0)
            pca_centers[i] = center

            # PCA using scikit-learn
            if len(cluster_points) >= 1:
                pca_model = PCA(n_components=3)
                pca_model.fit(cluster_points)

                pc_vectors = pca_model.components_       # shape (3,3)
                eigvals[i] = pca_model.explained_variance_  # shape (3,)
            else:
                # For a single point, use dummy axes
                pc_vectors = np.eye(3)
                eigvals[i] = np.array([0.0, 0.0, 0.0])

            # Assign principal axes
            pca_pc1[i] = pc_vectors[0]
            pca_pc2[i] = pc_vectors[1]
            pca_pc3[i] = pc_vectors[2]

            # Compute geometric features
            l1, l2, l3 = eigvals[i]

            if l1 > 0:
                linearity = (l1 - l2) / l1
                planarity = (l2 - l3) / l1
                scattering = l3 / l1
            else:
                linearity = 0.0
                planarity = 0.0
                scattering = 0.0

            features[i] = [linearity, planarity, scattering]

        return features, pca_centers, pca_pc1, pca_pc2, pca_pc3, eigvals
    
    def generate_cluster_ground_truth(self, cluster_labels, point_gt_labels, k):

        cluster_gt = np.zeros(k, dtype=int)

        for i in range(k):

            labels = point_gt_labels[cluster_labels == i]

            if len(labels) == 0:
                continue

            # majority vote
            values, counts = np.unique(labels, return_counts=True)
            cluster_gt[i] = values[np.argmax(counts)]

        return cluster_gt
    
    def svm(self, features, cluster_gt, eigvals):

        X = np.hstack([features, eigvals]) 

        # Scale before SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        svm_model = SVC(kernel='linear', gamma=0.01, C=10, class_weight="balanced")

        svm_model.fit(X_scaled, cluster_gt)

        predictions = svm_model.predict(X_scaled)

        return predictions

if __name__ == "__main__":
    pass