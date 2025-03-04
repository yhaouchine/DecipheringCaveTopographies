import numpy as np
import alphashape
import time
from process_cloud import import_cloud, display, pca_projection, pca_correction
from typing import Tuple


def calculate_alpha_shape(alpha: float, pts: list | np.ndarray) -> Tuple[any, float]:
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    The alpha shape is a generalization of the convex hull, allowing the identification
    of concave regions based on a given `alpha` parameter. A bigger `alpha` results
    in a more detailed and concave shape, while a smaller `alpha` generates a more convex shape.

    :param alpha: Controls the level of concavity (lower values = more concave, higher values = more convex).
        If `alpha` is too small, the shape may be disconnected or disappear.
    :param pts: A list or a NumPy array of shape (n, 2) representing `n` points in 2D space.

    :return: A tuple containing:
        - The computed alpha shape as a `Polygon`, `MultiPolygon`, or `None` (if the computation fails).
        - The time taken to compute the shape (in seconds).
    """
    start_time = time.perf_counter()
    a_shape = alphashape.alphashape(pts, alpha)
    end_time = time.perf_counter()
    t = end_time - start_time
    return a_shape, t


if __name__ == "__main__":
    point_cloud, point_cloud_name = import_cloud(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")

    # Reducing the cloud
    v_size_1 = 0.01
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)

    # Verify the number of points
    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to create a contour.")

    points_2d, pca_axes, mean, azimuth = pca_projection(points=points_reduced)

    print("PCA Axes:\n", pca_axes)
    print("Mean position:\n", mean)

    points_2d = pca_correction(pca_axes=pca_axes, points_2d=points_2d)

    alpha_shape, duration = calculate_alpha_shape(alpha=3.5, pts=points_2d)

    print(f"Contour computing time: {duration:.2f} secondes")

    # Check if the alpha shape was successfully generated
    if alpha_shape is None or not hasattr(alpha_shape, "exterior"):
        raise ValueError("Alpha-shape computation failed, try adjusting alpha.")

    x, y = alpha_shape.exterior.xy
    contour_2d = np.column_stack((x, y))

    display(pts=points_displayed, contour2d=contour_2d, projected_pts=points_2d, pca_axes=pca_axes)
