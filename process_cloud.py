# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "novembre 2024"

import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from open3d.cpu.pybind.geometry import PointCloud, Geometry, AxisAlignedBoundingBox
from pathlib import Path
from tkinter import simpledialog, messagebox, Tk
from sklearn.decomposition import PCA
from typing import Tuple

background_color = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
}


def import_cloud(pc_name: str, parent_folder: str) -> PointCloud | tuple:
    """
    Import point cloud with Open3D.

    @param pc_name: name of the point cloud file
    @param parent_folder: name of the folder containing the point cloud
    @return: The point cloud
    """
    pc_name = pc_name
    folder = parent_folder + "/"
    pc = o3d.io.read_point_cloud(folder + pc_name)
    return pc, pc_name


def o3d_visualizer(window_name: str, geom1: Geometry = None, geom2: Geometry = None, save: bool | None = None,
                   color_name: str = "white",
                   filename: Path | str | None = None) -> None | list:
    """
    Function to create a visualizer to display the data.

    @param save: Save the point cloud or not
    @param filename: name of the point cloud to be saved
    @param window_name: Name of the visualizer window
    @param geom1: First data to display
    @param geom2: Second data to display
    @param color_name: Color of the visualizer background
    @return: The point selected with shift + left click
    """
    save_folder = Path("saved_clouds")  # Folder containing the saved clouds
    save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
    back_color = background_color.get(color_name.lower(), [1.0, 1.0, 1.0])

    # If a bounding box is provided, use draw_geometries for combined visualization
    if geom1 and geom2:
        if isinstance(geom2, o3d.geometry.AxisAlignedBoundingBox):
            geom2.color = [1.0, 0.0, 0.0]  # Red color for the bounding box

        o3d.visualization.draw_geometries(
            [geom1, geom2],
            window_name=window_name,
            width=1280,
            height=800
        )
        return None  # No point selection in this mode

    # Use VisualizerWithEditing for single geometry (to allow picking points)
    visualizer = o3d.visualization.VisualizerWithEditing()
    visualizer.create_window(window_name=window_name, width=1280, height=800)
    visualizer.get_render_option().background_color = back_color

    # Visualizer manipulation infos
    print("Press X, Y or W key to change the camera angle according to X, Y and Z axes")
    print("Press K key to lock the view and enter the selection mode")
    print("Press C key to keep what is selected")
    print("Press S key to save the selection")
    print("Press F key to enter free-view mode")

    if geom1:
        visualizer.add_geometry(geom1)
    visualizer.run()

    root = Tk()
    root.withdraw()

    # Saved the point cloud if asked
    if geom1 and save:
        if not filename:
            filename = simpledialog.askstring("File Name", "Enter a name for the point cloud (e.g., 'cloud.ply'):")
            save_path = save_folder / filename
            o3d.io.write_point_cloud(str(save_path), geom1)
            print(f"Point cloud saved as: {save_path}")
    elif geom1 and save is None:
        user_choice = messagebox.askyesnocancel("Save Point Cloud", "Do you want to save the point cloud?")
        if user_choice is True:
            filename = simpledialog.askstring("File Name", "Enter a name for the point cloud (e.g., 'cloud.ply'):")
            save_path = save_folder / filename
            o3d.io.write_point_cloud(str(save_path), geom1)
            print(f"Point cloud saved as: {save_path}")

    picked_points = visualizer.get_picked_points()
    visualizer.destroy_window()
    return picked_points


def extract_cross_section(pc: PointCloud, position: np.ndarray, e: float | int) -> PointCloud:
    """
    Function to extract a cross-section from a point cloud

    @param pc: The initial point cloud from which the cross-section is extracted
    @param position: Position of the cross-section along the x-axis
    @param e: Thickness of the cross-section
    @return: The cross-section point cloud
    """

    points = np.asarray(pc.points)
    mask = (points[:, 0] > position - e / 2) & (points[:, 0] < position + e / 2)
    cut_points = points[mask]
    cut_pc = o3d.geometry.PointCloud()
    cut_pc.points = o3d.utility.Vector3dVector(cut_points)
    return cut_pc


def create_bounding_box(center_point: np.ndarray, size: float | int) -> AxisAlignedBoundingBox:
    """
    Function to create a bounding box
    
    @param center_point: Point representing the center of the bounding box
    @param size: Range of the bounding box centered on the center point
    @return: Bounding box object
    """

    min_bound = center_point - size / 2
    max_bound = center_point + size / 2
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


def compute_area(contour2d: np.ndarray) -> float:
    """
    Compute the area enclosed by a 2D contour using the Shoelace formula.

    @param contour2d:
    @return:
    """

    x, y = contour2d[:, 0], contour2d[:, 1]
    contour_area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1]))

    return contour_area


def display(pts: np.ndarray, contour2d: np.ndarray, projected_pts: np.ndarray, pca_axes: np.ndarray, azimuth: float,
            contour3d: np.ndarray = None) -> None:
    fig = plt.figure(figsize=(8, 8) if contour3d is None else (16, 8))

    # Adding a 3D plot if asked
    if contour3d is not None:
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, label="Point cloud")
        ax3d.plot(contour3d[:, 0], contour3d[:, 1], contour3d[:, 2], 'r--', linewidth=2.0, label="Contour (3D)")
        ax3d.set_title("3D Point Cloud & Contour")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.axis("equal")
        ax3d.legend()
        ax3d.set_box_aspect([1, 1, 1])
        ax2d = fig.add_subplot(122)
    else:
        ax2d = fig.add_subplot(111)

    # Calculate the area enclosed in the contour
    area = compute_area(contour2d)

    # PCA axes names
    pca_x_label = "PCA Axis 1"
    pca_y_label = "PCA Axis 2"

    # Fill the contour in the PCA plan
    polygon = Polygon(contour2d.tolist(), closed=True, facecolor='red', alpha=0.2, edgecolor='r', linewidth=2.0)
    ax2d.add_patch(polygon)

    ax2d.plot(contour2d[:, 0], contour2d[:, 1], 'r--', linewidth=2.0, label="Contour (In the PCA Plane)")
    ax2d.scatter(projected_pts[:, 0], projected_pts[:, 1], c='black', s=1, label="Projected points")

    # Position of the area value text
    text_x, _ = np.mean(contour2d, axis=0)
    _, text_y = np.max(contour2d, axis=0)
    ax2d.text(text_x, text_y, f"Area = {area:.2f} m²", fontsize=14, color='black', ha='center', va='top',
              bbox=dict(facecolor='white', alpha=0.6))

    ax2d.set_title("Contour in PCA Plane")
    ax2d.set_xlabel(f"Azimuth {azimuth:.1f}°")
    ax2d.set_ylabel("Perpendicular Axis")
    ax2d.legend()
    ax2d.axis("equal")
    plt.tight_layout()
    plt.show()


def pca_projection(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Projects a set of 3D points onto a 2D plane using Principal Component Analysis (PCA).

    This function is useful for analyzing cross-sections of 3D point clouds that are not aligned
    with the standard XYZ axes. It determines the principal directions of the point cloud and
    projects the points onto the first two principal components, effectively reducing the
    dimensionality from 3D to 2D.

    :param points: A NumPy array of shape (n, 3), representing `n` points in 3D space.

    :return: A tuple containing:
        - `projected_points` (np.ndarray of shape (n, 2)): The 2D coordinates of the projected points
          onto the PCA plane.
        - `pca_axes` (np.ndarray of shape (3, 3)): The three principal component vectors (each row is an eigenvector).
          The first two define the projection plane, and the third is the normal to this plane.
        - `mean` (np.ndarray of shape (3,)): The mean of the original points, used for centering before PCA.
    """

    # Compute the mean and center the points
    mean = np.mean(points, axis=0)
    centered_points = points - mean

    # Perform PCA to determine the principal directions
    pca = PCA(n_components=3)
    pca.fit(centered_points)

    # The normal to the best-fitting plane (third principal component)
    plane_normal = pca.components_[2]
    pca_axes = pca.components_

    # Compute the azimuth of the principal axis
    primary_axis = pca_axes[0]
    azimuth = np.degrees(np.arctan2(primary_axis[0], primary_axis[1]))

    # Project the points onto the first two principal components
    projected_points = centered_points @ pca_axes[:2].T

    return projected_points, pca.components_, mean, azimuth


def pca_correction(pca_axes, points_2d):
    """
    Corrects the orientation of the PCA axes so that the projected 2D points follow a consistent orientation.

    :param pca_axes: 3x3 matrix of the principal PCA axes
    :param points_2d: 2D point cloud projected in the PCA plane
    :return: corrected points_2d
    """
    reference_axes = np.eye(3)  # Matrice identité pour représenter les axes globaux (X, Y, Z)

    # Vérification et correction de l'orientation des axes PCA
    for i in range(2):  # Seulement les axes projetés (PCA X et PCA Y)
        dot_product = np.dot(pca_axes[:, i], reference_axes[i])  # Produit scalaire avec l'axe global
        if dot_product < 0:  # Si l'axe est inversé
            pca_axes[:, i] *= -1  # Inversion de l'axe PCA
            points_2d[:, i] *= -1  # Inversion des coordonnées projetées

    return points_2d


if __name__ == "__main__":

    # initializing GUI instance
    o3d.visualization.gui.Application.instance.initialize()

    point_cloud_name = "cave_res_1cm.ply"
    point_cloud = o3d.io.read_point_cloud("point_clouds/" + point_cloud_name)  # importing point cloud

    # Selecting points in the visualizer
    selected_indices = o3d_visualizer(window_name=point_cloud_name, geom1=point_cloud, save=False)
    if not selected_indices:
        print("No points selected. Exiting.")
        sys.exit()

    # Extracting the selected points
    selected_points = point_cloud.select_by_index(selected_indices)
    selected_points_coordinates = np.asarray(selected_points.points)

    # Position the cut on the point
    cut_position = selected_points_coordinates[0][0]
    thickness = simpledialog.askfloat("Tolerance of the cross-section", "Cross-section tolerance: ")
    cross_section_pc = extract_cross_section(point_cloud, cut_position, thickness)

    # Selecting point in the cross-section from which the Bounding Box is created
    center_index = o3d_visualizer(window_name=point_cloud_name + ": Cross-section", geom1=cross_section_pc)
    if not center_index:
        print("No point selected in cross-section. Exiting.")
        sys.exit()

# TODO:
#   Ajouter une valeur de tolérance pour l'épaisseur de la courbe, avec une valeur par défaut et une demande à l'utilisateur.
#   Possibilité de faire plusieurs coupes avec une distance entre les coupes constantes, ou des coupes à placer manuellement.
#   Projeter en 2D puis faire la réduction du nuage avec les voxels ?
#   Ne pas re projeter en 3D car ce n'est pas utile.
#   Documenter toutes les méthodes utilisées. Pourquoi préférer une méthode à une autre ?
#   Utiliser des class.
#   Utiliser une PCA pour projeter les points dans le cas ou la coupe n'est pas parfaitement alignée avec un axe.
#   Calculer aire de la surface afin de quantifier la performance du contour.
