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

    def k_means_algo(self, points, k):
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
    
    def remove_floor(self, points, cell_size=0.5, initial_window=1, max_window=8,
                            slope=0.1, initial_elevation_threshold=0.1, max_elevation_threshold=0.5,
                            classification_threshold=0.1, min_points_above=3):
        """
        classification_threshold (float): tight fixed threshold used ONLY for final point 
                                        classification against the ground surface. 
                                        Keep this small — it is the maximum height above 
                                        the reconstructed ground a point can be to be 
                                        called floor. Rooftops will exceed this since they
                                        sit well above the reconstructed ground surface.
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        cols = int(np.ceil((x_max - x_min) / cell_size)) + 1
        rows = int(np.ceil((y_max - y_min) / cell_size)) + 1

        col_indices = ((x - x_min) / cell_size).astype(int)
        row_indices = ((y - y_min) / cell_size).astype(int)

        # Build minimum Z grid
        min_z_grid = np.full((rows, cols), np.inf)
        for idx in range(len(points)):
            r, c = row_indices[idx], col_indices[idx]
            if z[idx] < min_z_grid[r, c]:
                min_z_grid[r, c] = z[idx]

        ground_surface = np.where(np.isinf(min_z_grid), np.nan, min_z_grid)

        for c in range(cols):
            col = ground_surface[:, c]
            mask = np.isnan(col)
            if mask.all():
                continue
            indices = np.where(~mask)[0]
            ground_surface[:, c] = np.interp(np.arange(rows), indices, col[indices])

        from scipy.ndimage import minimum_filter, maximum_filter

        window_size = initial_window
        prev_surface = ground_surface.copy()

        while window_size <= max_window:
            eroded = minimum_filter(prev_surface, size=window_size)
            opened = maximum_filter(eroded, size=window_size)

            elevation_threshold = min(
                initial_elevation_threshold + slope * window_size * cell_size,
                max_elevation_threshold
            )
            new_surface = np.where(
                (prev_surface - opened) < elevation_threshold,
                opened,
                prev_surface
            )
            prev_surface = new_surface
            window_size *= 2

        ground_surface_final = prev_surface

        # Use a fixed tight threshold for classification, not the scaled PMF threshold
        # This is the key change — rooftops sit far above ground_surface_final,
        # so a small classification_threshold cleanly separates them from the true floor
        point_ground_z = ground_surface_final[row_indices, col_indices]
        elevation_above_ground = z - point_ground_z
        is_floor_candidate = elevation_above_ground < classification_threshold

        # Vertical density check — reject candidates with many points above (bases of objects)
        cell_flat = row_indices * cols + col_indices
        points_above_count = np.zeros(rows * cols, dtype=int)
        np.add.at(points_above_count, cell_flat[~is_floor_candidate], 1)
        has_objects_above = points_above_count[cell_flat] > min_points_above

        is_floor = is_floor_candidate & ~has_objects_above

        floor_removed = points[~is_floor]
        floor_points = points[is_floor]

        print(f"PMF v4 removed {is_floor.sum()} floor points, {len(floor_removed)} points remaining.")
        return floor_removed, floor_points

    def k_means(self, points, k):
        points_no_floor, floor_points = self.remove_floor(points)
        centroids, assignment = self.k_means_algo(points_no_floor, k)
        # centroids, assignment = self.k_means_algo(points, k)
        return centroids, assignment, points_no_floor

    def pca(self):
        pass

    def svm(self):
        pass

if __name__ == "__main__":
    pass