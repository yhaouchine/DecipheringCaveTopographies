import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from process_cloud import import_cloud
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


def display(pts: np.ndarray, contour: np.ndarray, vox_size1: float, vox_size2: float) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original point cloud
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, label=f"Point cloud (voxel size = {vox_size2})")

    # Plot the contour
    ax.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'r--', linewidth=1.5,
            label=f"Concave hull (voxel size = {vox_size1})")

    ax.set_title("3D Point Cloud & Contour")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("equal")
    plt.tight_layout()
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":

    point_cloud, _ = import_cloud(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")

    v_size_1: float = 2.0
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    v_size_2: float = 1.0
    displayed_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_2)
    points_displayed = np.asarray(displayed_point_cloud.points)

    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to generate a contour.")

    points_2d = points_reduced[:, 1:3]  # YZ Projection

    # calculate the concave hull in 2D
    concavity = 1.5
    length_threshold = 1.0
    hull_2d = compute_concave_hull_2d(points_2d, concavity, length_threshold)

    # Build a KDTree to project the points in 3D
    points_kdtree = KDTree(points_2d)
    _, indices_3d = points_kdtree.query(hull_2d)

    # Build the 3D contour with the original X coordinates of the points
    contour_3d = points_reduced[indices_3d]

    display(pts=points_displayed, contour=contour_3d, vox_size1=v_size_1, vox_size2=v_size_2)
