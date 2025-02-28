import numpy as np
import alphashape
from process_cloud import import_cloud, display, compute_area, project_points_pca, correct_pca_orientation


def calculate_alpha_shape(alpha: float, pts: list | np.ndarray) -> any:
    a_shape = alphashape.alphashape(pts, alpha)
    return a_shape


if __name__ == "__main__":
    point_cloud, point_cloud_name = import_cloud(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")

    # Reducing the cloud
    v_size_1 = 0.1
    reduced_point_cloud = point_cloud.voxel_down_sample(voxel_size=v_size_1)
    points_reduced = np.asarray(reduced_point_cloud.points)

    points_displayed = np.asarray(point_cloud.points)

    # Verify the number of points
    if points_reduced.shape[0] < 3:
        raise ValueError("Not enough points to create a contour.")

    points_2d, pca_axes, mean = project_points_pca(points=points_reduced)

    print("PCA Axes:\n", pca_axes)
    print("Mean position:\n", mean)

    points_2d = correct_pca_orientation(pca_axes=pca_axes, points_2d=points_2d)

    # Calculate the alpha shape
    alpha_shape = calculate_alpha_shape(alpha=0.5, pts=points_2d)

    # Check if the alpha shape was successfully generated
    if alpha_shape is None or not hasattr(alpha_shape, "exterior"):
        raise ValueError("Alpha-shape computation failed, try adjusting alpha.")

    x, y = alpha_shape.exterior.xy
    contour_2d = np.column_stack((x, y))

    display(pts=points_displayed, contour2d=contour_2d, pts_2d=points_2d, pca_axes=pca_axes)
