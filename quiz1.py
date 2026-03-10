import argparse
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import open3d as o3d
import polyscope as ps
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
from quiz1_algorithms import Algorithms


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ply_point_cloud(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load a point cloud from a .ply file.
    Returns points and binary ground truth labels derived from color (1=ground, 0=other).
    """
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    if points.size == 0:
        raise ValueError(f"No points found in {path}")

    labels = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        if colors.shape == points.shape:
             # Ground color reference (Brown)
             ground_color_ref = np.array([0.6, 0.4, 0.1])
             # Check for exact equality
             is_ground = np.all(np.isclose(colors, ground_color_ref, atol=1e-2), axis=1)
             labels = is_ground.astype(int)

    return points, labels


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize(points: np.ndarray, labels: np.ndarray, cluster_labels: np.ndarray, k: int, 
              pca_centers, pca_pc1, pca_pc2, pca_pc3, cluster_gt, point_predictions) -> None:
    """
    Set up Polyscope serialization.
    """
    print("Visualizing results in Polyscope...")
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")


    # 1. Register main point cloud
    cloud = ps.register_point_cloud("Processed Point Cloud", points, radius=0.0015)
    cloud.set_point_render_mode("quad")
    cloud.add_scalar_quantity("Elevation", points[:, 2], enabled=True)

    # 2. Add ground truth labels
    cloud.add_scalar_quantity("Ground truth data", labels, enabled=False)
    
    # 3. Add colors for clusters
    cloud.add_scalar_quantity(
        "KMeans Clusters",
        cluster_labels,
        enabled=True
    )
    # 4. Register PCA visualization (Cluster Centers and Principal Components)
    pca_cloud = ps.register_point_cloud("Cluster PCA Centers", pca_centers, radius=0.02)
    pca_cloud.add_vector_quantity( "PC1 (Major)", pca_pc1, enabled=True, color=(1,0,0), vectortype="ambient")
    pca_cloud.add_vector_quantity( "PC2 (Minor)", pca_pc2, enabled=False, color=(0,1,0), vectortype="ambient")
    pca_cloud.add_vector_quantity( "PC3 (Normal)", pca_pc3, enabled=False, color=(0,0,1), vectortype="ambient")
    # 5. Add SVM Predictions
    cloud.add_scalar_quantity( "Cluster Ground Truth", cluster_gt[cluster_labels], enabled=False)
    cloud.add_scalar_quantity("SVM Predictions", point_predictions, enabled=True)

    ps.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    algo = Algorithms()
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation pipeline (K-Means -> PCA -> SVM)")
    parser.add_argument("path", nargs="?", default="airport_downsample.ply")
    parser.add_argument("-k", "--clusters", type=int, default=20, help="Number of k-means clusters (default: 6)")
    args = parser.parse_args()

    # 1. Load Data
    ## returns points and labels
    points, point_gt_labels = load_ply_point_cloud(args.path)
    print(f"Loaded {len(points)} points from {args.path}")

    # You might want to create functions for each of the following points to keep the code clean

    # 2. Clustering

    # sil_scores = []

    # for k in range(2, 10):
    #     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    #     labels = kmeans.fit_predict(points)
    #     score = silhouette_score(points, labels)
    #     sil_scores.append(score)

    # plt.plot(range(2, 10), sil_scores, marker='o')
    # plt.xlabel('Number of clusters k')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Analysis for Optimal k')
    # plt.show()

    centroids, cluster_labels = algo.k_means(points, args.clusters)

    # if os.path.exists("cluster_labels.npy"):
    #     print("Loading saved cluster data...")
    #     cluster_labels = np.load("cluster_labels.npy")
    #     centroids = np.load("cluster_centroids.npy")
    # else:
    #     print("Running K-Means clustering...")
    #     centroids, cluster_labels = algo.k_means(points, args.clusters)
    #     np.save("cluster_labels.npy", cluster_labels)
    #     np.save("cluster_centroids.npy", centroids)

    # 3. Feature Extraction (PCA)
    features, pca_centers, pca_pc1, pca_pc2, pca_pc3 = algo.extract_cluster_features(
        points,
        cluster_labels,
        args.clusters
    )
    # 4. Ground Truth Generation (for training SVM)
    cluster_gt = algo.generate_cluster_ground_truth( cluster_labels, point_gt_labels, args.clusters)
    # 5. SVM Classification
    print("Cluster GT labels:", cluster_gt)
    print("Unique classes:", np.unique(cluster_gt))
    predictions = algo.svm(features, cluster_gt)
    point_predictions = predictions[cluster_labels]
    # 6. Visualization
    visualize(points, point_gt_labels, cluster_labels, args.clusters,  
                pca_centers, pca_pc1, pca_pc2, pca_pc3, cluster_gt, point_predictions)


if __name__ == "__main__":
    main()