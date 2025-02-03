# -*- coding:utf-8 -*-

import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from process_cloud import o3d_visualizer
from tkinter import simpledialog, messagebox, Tk


def create_ellipsoid(pc: o3d.geometry.PointCloud, selected_i: list[int], resolution: int = 15,
                     a_axis: float = 1.0,
                     b_axis: float = 1.0,
                     c_axis: float = 1.0
                     ) -> tuple[o3d.geometry.LineSet, np.ndarray, list[float]]:
    """
    Creates an ellipsoid.

    @param pc: The point cloud object.
    @param selected_i: Indices of the two points selected by the user (top and bottom).
    @param resolution: Resolution of the ellipsoid mesh.
    @param a_axis: Length of a-axis (x-axis) of the ellipsoid
    @param b_axis: Length of b-axis (y-axis) of the ellipsoid
    @param c_axis: Length of c-axis (z-axis) of the ellipsoid
    @return: Open3D LineSet representing the wireframe ellipsoid, the center point, and the axes lengths.
    """

    if len(selected_i) == 2:

        # Get the coordinates of the selected points
        z_top = np.asarray(pc.points)[selected_i[0]]
        z_bottom = np.asarray(pc.points)[selected_i[1]]

        # Calculate the center of the ellipsoid (mean of the two selected points)
        center_point = (z_top + z_bottom) / 2
        # Calculate the vertical distance between the two points and define initial axes lengths
        c_axis = np.linalg.norm(z_top - z_bottom)
    elif len(selected_i) == 1:
        center_point = np.asarray(pc.points)[selected_i[0]]

    # Define axes lengths
    l_axes = [a_axis / 2, b_axis / 2, c_axis / 2]  # Semi-axes lengths

    # Create a unit sphere and transform it into an ellipsoid
    meshed_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    meshed_ellipsoid.scale(1.0, meshed_ellipsoid.get_center())  # Ensure unit sphere scaling
    scaling_matrix = np.diag([l_axes[0], l_axes[1], l_axes[2], 1.0])
    meshed_ellipsoid.transform(scaling_matrix)  # Scale axes
    meshed_ellipsoid.translate(center_point)  # Translate to center

    # Convert the mesh to wireframe using edges
    edges = []
    for triangle in np.asarray(meshed_ellipsoid.triangles):
        edges.append([triangle[0], triangle[1]])
        edges.append([triangle[1], triangle[2]])
        edges.append([triangle[2], triangle[0]])

    edges = np.array(edges)
    unique_edges = np.unique(edges, axis=0)  # Remove duplicate edges
    lines = unique_edges.tolist()

    # Extract points from the mesh
    points = np.asarray(meshed_ellipsoid.vertices)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # Red wireframe

    return line_set, center_point, l_axes


def filter_points_in_ellipsoid(pc: PointCloud, p_center: np.ndarray, l_axes: list) -> PointCloud:
    """
    Filters points within an ellipsoid defined by center and semi-axes.

    @param pc: Open3D point cloud object.
    @param p_center: Center point of the ellipsoid (x, y, z).
    @param l_axes: Lengths of the semi-axes (a, b, c).
    @return: Filtered point cloud.
    """
    points = np.asarray(pc.points)
    relative_positions = points - p_center
    normalized_distances = (
            (relative_positions[:, 0] / l_axes[0]) ** 2
            + (relative_positions[:, 1] / l_axes[1]) ** 2
            + (relative_positions[:, 2] / l_axes[2]) ** 2
    )
    mask = normalized_distances > 1.0  # Points outside ellipsoid
    filtered_points = points[mask]
    filtered_pc = o3d.geometry.PointCloud()
    filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pc


if __name__ == "__main__":
    point_cloud_name = "section_with_irrelevant_pts_2.ply"
    point_cloud = o3d.io.read_point_cloud("saved_clouds/" + point_cloud_name)

    while True:
        # Select points
        print("Please select one or two points: ")
        selected_indices = o3d_visualizer(window_name=point_cloud_name, geom1=point_cloud, save=False)

        if not selected_indices:
            print("No points selected. The cloud is considered clean.")
            break

        # Create ellipsoid based on the two points selected
        ellipsoid, center, axes_lengths = create_ellipsoid(point_cloud, selected_indices)

        while True:
            # Visualize the point cloud with the ellipsoid
            o3d_visualizer(window_name="Point Cloud with Ellipsoid", geom1=point_cloud, geom2=ellipsoid)

            root = Tk()
            root.withdraw()
            user_input = messagebox.askyesnocancel("Validation", "Does the ellipsoid fit the points?")

            if user_input is True:
                print("Ellipsoid confirmed. Removing points within...")
                break

            elif user_input is False:
                try:
                    new_a_axis = simpledialog.askfloat("Input", "Enter new value for a-axis length:")
                    new_b_axis = simpledialog.askfloat("Input", "Enter new value for b-axis length:")
                    new_c_axis = 1.0
                    if len(selected_indices) == 1:
                        new_c_axis = simpledialog.askfloat("Input", "Enter new value for c-axis length:")

                    ellipsoid, center, axes_lengths = create_ellipsoid(
                        point_cloud,
                        selected_indices,
                        a_axis=new_a_axis,
                        b_axis=new_b_axis,
                        c_axis=new_c_axis,
                        resolution=15
                    )

                except ValueError:
                    print("Invalid input. Please enter numeric values.")
            elif user_input is None:
                print("Operation canceled")
                exit()

        # Filter points inside the ellipsoid
        filtered_cloud = filter_points_in_ellipsoid(point_cloud, p_center=center, l_axes=axes_lengths)
        point_cloud = filtered_cloud

    user_input_2 = messagebox.askyesnocancel("Save", "Do you want to save the cloud?")
    if user_input_2 is True:
        saved_point_cloud_name = simpledialog.askstring("Filename",
                                                        "Name of the file with extension (e.g. Cloud.ply): ")
        print("Saving filtered point cloud...")
        o3d.io.write_point_cloud("saved_clouds/filtered_" + saved_point_cloud_name, point_cloud)
        print("Filtered point cloud saved as " + saved_point_cloud_name + "'.")
    elif user_input_2 is False:
        print("Process complete.")
    elif user_input_2 is None:
        print("Operation canceled")
        exit()
