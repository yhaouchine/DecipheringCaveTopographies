import numpy as np
import time
from typing import Tuple
from scipy.spatial import KDTree
from process_cloud import import_cloud, display, pca_projection, pca_correction
from concave_hull import concave_hull


def compute_concave_hull_2d(pts_2d: np.ndarray, c: float = 1.0, l_threshold: float = 0.0) -> Tuple[np.ndarray, float]:
    """
    Compute the concave hull (alpha shape) for a set of 2D points.

    The concave hull is a polygon that better approximates the shape of a point cloud than a convex hull,
    allowing concavities based on the `concavity` parameter.

    :param pts_2d: A NumPy array of shape (n, 2) representing `n` points in 2D space.
    :param c: Concavity coefficient (small values = highly detailed concave shape, larger values = more convex shape).
        A high value may result in a convex hull.
    :param l_threshold: Length threshold below which segments are ignored in the hull construction.

    :return: A tuple containing:
        - `hull`: A NumPy array of shape (m, 2) representing the ordered points forming the concave polygon.
        - `time`: The time taken to compute the concave hull (in seconds).
    """
    start_time = time.perf_counter()
    hull = concave_hull(pts_2d, concavity=c, length_threshold=l_threshold)
    hull = np.array(hull)

    if not np.allclose(hull[0], hull[-1]):      # Ensure the hull is closed
        hull = np.vstack((hull, hull[0]))

    end_time = time.perf_counter()
    t = end_time - start_time
    return hull, t


if __name__ == "__main__":

    point_cloud, _ = import_cloud(pc_name="cross_section_2_clean.ply", parent_folder="saved_clouds")

    v_size_1: float = 0.01
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)

    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to generate a contour.")

    points_2d, pca_axes, mean, azimuth = pca_projection(points=points_reduced)

    print("PCA Axes:\n", pca_axes)
    print("Mean position:\n", mean)

    points_2d = pca_correction(pca_axes=pca_axes, points_2d=points_2d)

    # calculate the concave hull in 2D
    concavity = 1.0
    length_threshold = 0.15
    hull_2d, duration = compute_concave_hull_2d(points_2d, concavity, length_threshold)

    print(f"Contour computing time: {duration:.2f} secondes")

    # Build a KDTree to project the points in 3D
    points_kdtree = KDTree(points_2d)
    _, indices_3d = points_kdtree.query(hull_2d)

    # Build the 3D contour with the original X coordinates of the points
    contour_3d = points_reduced[indices_3d]

    display(pts=points_displayed, contour2d=hull_2d, projected_pts=points_2d, pca_axes=pca_axes, azimuth=azimuth)
