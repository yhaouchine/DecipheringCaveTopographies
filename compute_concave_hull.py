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


def display(pts: np.ndarray, contour: np.ndarray, contour_2d: np.ndarray, vox_size1: float, vox_size2: float) -> None:
    # Affichage 3D et 2D côte à côte dans une figure
    fig = plt.figure(figsize=(16, 8))

    # Affichage 3D
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1,
                 label=f"Point cloud (voxel size = {vox_size2})")
    ax3d.plot(contour[:, 0], contour[:, 1], contour[:, 2], 'r--', linewidth=2.0,
              label=f"Concave hull (voxel size = {vox_size1})")
    ax3d.set_title("3D Point Cloud & Contour")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.legend()
    ax3d.axis("equal")

    # Affichage 2D (projection YZ)
    ax2d = fig.add_subplot(122)
    ax2d.plot(contour_2d[:, 0], contour_2d[:, 1], 'r--', linewidth=2.0, label="Concave hull (YZ projection)")
    ax2d.scatter(contour_2d[:, 0], contour_2d[:, 1], c='red', s=10)
    ax2d.set_title("2D Concave Hull (YZ Projection)")
    ax2d.set_xlabel("Y")
    ax2d.set_ylabel("Z")
    ax2d.legend()
    ax2d.axis("equal")

    plt.tight_layout()
    plt.show()


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

    display(pts=points_displayed, contour=contour_3d, contour_2d=hull_2d, vox_size1=v_size_1, vox_size2=v_size_2)
