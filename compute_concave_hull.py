import numpy as np
import time
from typing import Tuple
from process_cloud import import_cloud, display, pca_projection
from concave_hull import concave_hull


def compute_concave_hull_2d(pts_2d: np.ndarray, c: float = 1.0, l_threshold: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Compute the concave hull (alpha shape) for a set of 2D points.

    :param pts_2d: A NumPy array of shape (n, 2) representing `n` points in 2D space.
    :param c: Concavity coefficient (small values = highly detailed concave shape, larger values = more convex shape).
    :param l_threshold: Length threshold below which segments are ignored in the hull construction.

    :return: A tuple containing:
        - `hull`: A NumPy array of shape (m, 2) representing the ordered points forming the concave polygon.
        - `time`: The time taken to compute the concave hull (in seconds).
    """
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
    v_size_1: float = 0.01
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)


    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to generate a contour.")

    points_2d, pca_axes, mean = pca_projection(points_3d=points_reduced, diagnosis=True, display=True)

    # Compute the concave hull in 2D
    concavity = 1.0
    length_threshold = 0.15
    hull_2d, duration = compute_concave_hull_2d(points_2d, concavity, length_threshold)

    print(f"Contour computing time: {duration:.2f} secondes")

    display(pts=points_displayed, contour2d=hull_2d, projected_pts=points_2d, pca_axes=pca_axes)