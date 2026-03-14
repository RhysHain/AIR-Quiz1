# All algorithims are here

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from scipy.linalg import svd
import open3d as o3d
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

class Algorithms:

    def __init__(self):
        self.rng = np.random.default_rng(4)

    def k_means(self, points, k):
        scaled = StandardScaler()
        scaled = scaled.fit_transform(points)
        kmeans = KMeans(n_clusters=k)
        
        kmeans.fit(scaled)
        
        centroids = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        
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
    
    # Function that allows for searching for best svm variables
    # def svm(self, features, cluster_gt, eigvals):
    #     # 1. Prepare and Scale
    #     X = np.hstack([features, eigvals])
    #     scaler = StandardScaler()
    #     X_scaled = scaler.fit_transform(X)

    #     # 2. Define the Search (This is Option 3)
    #     param_grid = {
    #         'C': [0.1, 1, 10, 100],
    #         'gamma': [1, 0.1, 0.01, 'scale'],
    #         'kernel': ['rbf'] # RBF is usually superior for these features
    #     }

    #     # 3. Run the search
    #     # cv=5 means it tests each setting 5 times on different data slices
    #     grid = GridSearchCV(
    #         SVC(class_weight='balanced'), 
    #         param_grid, 
    #         cv=5, 
    #         scoring='f1_macro' # Optimizes for both classes equally
    #     )
        
    #     print("Searching for best SVM parameters...")
    #     grid.fit(X_scaled, cluster_gt)
        
    #     # 4. Use the best version found
    #     best_model = grid.best_estimator_
    #     print(f"Best settings found: {grid.best_params_}")

    #     # 5. Predict
    #     # You can now use cross_val_predict with the 'best_model' 
    #     # or just predict on the current set
    #     predictions = cross_val_predict(best_model, X_scaled, cluster_gt, cv=10)

    #     return predictions
    
    def svm(self, features, cluster_gt, eigvals):
        # 1. Combine and Scale Features
        X = np.hstack([features, eigvals])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Setup the SVM
        # Using 'rbf' often handles the 10% held-out data better than 'linear'
        svm_model = SVC(kernel='linear', C=1, class_weight='balanced', gamma=0.01)
        
        # 3. Perform Cross-Validation
        # cv=10 splits the data into 10 parts (90% train, 10% test)
        # predictions will be a (k,) array where each prediction was made 
        # when that specific cluster was in the 10% "unseen" group.
        predictions = cross_val_predict(svm_model, X_scaled, cluster_gt, cv=10)


        
        # 4. Calculate the ground vs not-ground percentages
        # Assuming 0 = Ground, 1 = Not Ground
        # cm = confusion_matrix(cluster_gt, predictions, labels=[0, 1])
        
        # ground_acc = (cm[0, 0] / cm[0, :].sum()) * 100
        # not_ground_acc = (cm[1, 1] / cm[1, :].sum()) * 100
        
        # print(f"Cross-Validated Ground Accuracy:     {ground_acc:.2f}%")
        # print(f"Cross-Validated Not Ground Accuracy: {not_ground_acc:.2f}%")

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(svm_model, X_scaled, cluster_gt, cv=10)

        print(f"10-Fold Cross-Validation Results:")
        print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
        print(f"Standard Deviation: {scores.std() * 100:.2f}%")

        return predictions

if __name__ == "__main__":
    pass