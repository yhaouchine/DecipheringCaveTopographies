import numpy as np
import matplotlib.pyplot as plt
import alphashape
from matplotlib.patches import Polygon
from process_cloud import import_cloud


def calculate_alpha_shape(alpha: float, pts: list | np.ndarray) -> any:
    a_shape = alphashape.alphashape(pts, alpha)
    return a_shape


def compute_area(contour2d: np.ndarray) -> float:
    """
    Compute the area enclosed by a 2D contour using the Shoelace formula.

    @param contour2d:
    @return:
    """

    x, y = contour2d[:, 0], contour2d[:, 1]
    contour_area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1]))

    return contour_area


def display(pts: np.ndarray, contour2d: np.ndarray, contour3d: np.ndarray = None) -> None:
    fig = plt.figure(figsize=(8, 8) if contour3d is None else (16, 8))

    if contour3d is not None:
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, label="Point cloud")
        ax3d.plot(contour3d[:, 0], contour3d[:, 1], contour3d[:, 2], 'r--', linewidth=2.0, label="Convex hull")
        ax3d.set_title("3D Point Cloud & Contour")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.legend()
        ax3d.axis("equal")
        ax2d = fig.add_subplot(122)
    else:
        ax2d = fig.add_subplot(111)

    area = compute_area(contour2d)

    # Fill the contour with a transparent color
    polygon = Polygon(contour2d, closed=True, facecolor='red', alpha=0.1, edgecolor='r', linewidth=2.0)
    ax2d.add_patch(polygon)

    ax2d.plot(contour2d[:, 0], contour2d[:, 1], 'r--', linewidth=2.0, label="Concave hull (YZ projection)")
    ax2d.scatter(pts[:, 1], pts[:, 2], c='black', s=1)

    text_x, _ = np.mean(contour2d, axis=0)
    _, text_y = np.max(contour2d, axis=0)
    ax2d.text(text_x, text_y, f"Area = {area:.2f} m²", fontsize=14, color='black', ha='center', va='top',
              bbox=dict(facecolor='white', alpha=0.6))

    ax2d.set_title("2D Concave Hull (YZ Projection)")
    ax2d.set_xlabel("Y")
    ax2d.set_ylabel("Z")
    ax2d.legend()
    ax2d.axis("equal")

    plt.tight_layout()
    plt.show()


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

    # Project the cloud in the 2D Y-Z plan
    points_2d = points_reduced[:, 1:3]

    # Calculate the alpha shape
    alpha_shape = calculate_alpha_shape(alpha=8.0, pts=points_2d)

    # Check if the alpha shape was successfully generated
    if alpha_shape is None or not hasattr(alpha_shape, "exterior"):
        raise ValueError("Alpha-shape computation failed, try adjusting alpha.")

    x, y = alpha_shape.exterior.xy
    contour_2d = np.column_stack((x, y))

    display(pts=points_displayed, contour2d=contour_2d)
