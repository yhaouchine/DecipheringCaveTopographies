import numpy as np
from typing import Tuple
from process_cloud import import_cloud, display, pca_projection
from concave_hull import concave_hull


def compute_concave_hull_2d(pts_2d: np.ndarray, c: float = 1.0, l_threshold: float = 0.0) -> Tuple[np.ndarray, float]:
    """
       Compute the concave hull for a set of 2D points using a K-Nearest Neighbors (KNN)
       approach based on the Concaveman algorithm.

       The concave hull is a polygon that more accurately follows the natural boundary of a point cloud
       than the convex hull. Unlike the convex hull, which is the smallest convex polygon that encloses all
       the points, the concave hull allows for indentations and concavities, providing a closer approximation
       to the true shape of the data.

       This implementation follows the principles described in the Concaveman algorithm
       (see: https://github.com/mapbox/concaveman), and uses two main parameters to control the level
       of detail of the resulting hull:

         - Concavity coefficient (c):
               * Lower values produce a more detailed, highly concave shape.
               * Higher values produce a smoother, more convex-like shape.
         - Length threshold (l_threshold):
               * Specifies a minimum edge length below which segments are ignored in the hull construction.
               * This helps to filter out spurious edges that may result from noise in the data.

       The algorithm typically works as follows:
         1. An initial boundary is computed (often starting from the convex hull).
         2. For each edge of the current boundary, the algorithm considers candidate points using a k-nearest
            neighbors criterion.
         3. It selects a candidate point that minimizes the turning angle relative to the current segment while
            avoiding edge crossings (using geometric tests such as the orientation test).
         4. Segments with a length shorter than the specified threshold (l_threshold) are discarded, which reduces
            the effect of noise.
         5. The process continues iteratively until a closed polygon is formed that closely approximates the true
            boundary of the point cloud.

       Parameters:
       ----------
       pts_2d : np.ndarray
           A NumPy array of shape (n, 2) containing n points in 2D space, where each row represents the (x, y)
           coordinates of a point.
       c : float, optional (default=1.0)
           The concavity coefficient controlling the level of detail of the hull:
             - Lower values yield a more detailed, concave shape.
             - Higher values yield a smoother, more convex shape.
       l_threshold : float, optional (default=0.0)
           The minimum edge length below which segments are ignored during the hull construction,
           which helps filter out edges caused by noise.

       Returns:
       -------
        A tuple containing:
            - hull: A NumPy array of shape (m, 2) of the ordered vertices of the concave hull polygon.
            - time: A float representing the computation time in seconds.
   """

    import time

    start_time = time.perf_counter()
    hull = concave_hull(pts_2d, concavity=c, length_threshold=l_threshold)
    hull = np.array(hull)

    if not np.allclose(hull[0], hull[-1]):  # Ensure the hull is closed
        hull = np.vstack((hull, hull[0]))

    end_time = time.perf_counter()
    t = end_time - start_time
    return hull, t


if __name__ == "__main__":
    # Import point cloud
    point_cloud, _ = import_cloud(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")

    # Downsample the point cloud
    v_size_1: float = 0.5
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)

    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to generate a contour.")

    points_2d, pca_axes, mean = pca_projection(points_3d=points_reduced, diagnosis=True, visualize=True)

    # Compute the concave hull in 2D
    concavity = 1.0
    length_threshold = 0.15
    hull_2d, duration = compute_concave_hull_2d(points_2d, concavity, length_threshold)

    print(f"Contour computing time: {duration:.2f} secondes")

    display(pts=points_displayed, contour2d=hull_2d, projected_pts=points_2d)
