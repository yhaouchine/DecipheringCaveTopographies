import numpy as np
from scipy.spatial import KDTree
from process_cloud import import_cloud, display, compute_area
from concave_hull import concave_hull


def compute_concave_hull_2d(pts_2d: np.ndarray, c: float = 1.0, l_threshold: float = 0.0) -> np.ndarray:
    """
    calculate the concave hull for a set of 2d points.

    :param pts_2d: np.ndarray of shape (n, 2)
    :param c: coefficient of concavity (1 = detailed concave shape, infinite = convex shape)
    :param l_threshold: length threshold below which the segments are not considered any more
    :return: hull, a np.ndarray of shape (m, 2) describing the concave polygon
    """
    hull = concave_hull(pts_2d, concavity=c, length_threshold=l_threshold)
    hull = np.array(hull)

    if not np.allclose(hull[0], hull[-1]):
        hull = np.vstack((hull, hull[0]))
    return hull


if __name__ == "__main__":

    point_cloud, _ = import_cloud(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")

    v_size_1: float = 0.001
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)

    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to generate a contour.")

    points_2d = points_reduced[:, 1:3]  # YZ Projection

    # calculate the concave hull in 2D
    concavity = 0.01
    length_threshold = 0.05
    hull_2d = compute_concave_hull_2d(points_2d, concavity, length_threshold)

    # Build a KDTree to project the points in 3D
    points_kdtree = KDTree(points_2d)
    _, indices_3d = points_kdtree.query(hull_2d)

    # Build the 3D contour with the original X coordinates of the points
    contour_3d = points_reduced[indices_3d]

    display(pts=points_displayed, contour2d=hull_2d)
