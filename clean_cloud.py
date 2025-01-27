# -*- coding:utf-8 -*-

import sys
import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud

from process_cloud import o3d_visualizer


def create_ellipsoid_from_two_points(pc: PointCloud, selected_i: list, wireframe: bool = False,
                                     resolution: int = 30) -> tuple | list | np.ndarray:
    """
    Creates an ellipsoid based on two selected points: one at the top and one at the bottom of the feature.

    @param pc: The point cloud object.
    @param selected_i: Indices of the two points selected by the user (top and bottom).
    @param wireframe: Whether to display the ellipsoid as wireframe.
    @param resolution: Resolution of the ellipsoid mesh.
    @return: Open3D geometry (TriangleMesh for solid or LineSet for wireframe).
    """
    if len(selected_indices) != 2:
        print("You need to select exactly two points.")
        sys.exit()

    # Get the coordinates of the selected points
    point_top = np.asarray(pc.points)[selected_i[0]]
    point_bottom = np.asarray(pc.points)[selected_i[1]]

    # Calculate the center of the ellipsoid (mean of the two selected points)
    center_point = (point_top + point_bottom) / 2

    # Calculate the vertical distance between the two points and define axes lengths
    vertical_distance = np.linalg.norm(point_top - point_bottom)
    l_axes = [1.0, 1.0, vertical_distance / 2]  # You can adjust these values to suit your needs

    # Create the ellipsoid using the calculated parameters
    meshed_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    meshed_ellipsoid.scale(1.0, meshed_ellipsoid.get_center())  # Ensure unit sphere scaling
    meshed_ellipsoid.transform(np.diag([l_axes[0], l_axes[1], l_axes[2], 1.0]))  # Scale axes
    meshed_ellipsoid.translate(center_point)  # Translate to center

    if wireframe:
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
    else:
        return meshed_ellipsoid, center_point, l_axes


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
    return filtered_cloud


if __name__ == "__main__":
    # Load point cloud
    point_cloud_name = "section_with_irrelevant_pts_2.ply"
    point_cloud = o3d.io.read_point_cloud("saved_clouds/" + point_cloud_name)

    # Select two points (top and bottom)
    print("Please select two points: one at the top and one at the bottom of the feature.")
    selected_indices = o3d_visualizer(window_name=point_cloud_name, geom1=point_cloud, save=False)

    # Create ellipsoid based on the two points selected
    ellipsoid, center, axes_lengths = create_ellipsoid_from_two_points(point_cloud, selected_indices, wireframe=True)

    # Visualize the point cloud with the ellipsoid
    o3d_visualizer(window_name="Point Cloud with Ellipsoid", geom1=point_cloud, geom2=ellipsoid)

    # Filter points inside the ellipsoid
    filtered_cloud = filter_points_in_ellipsoid(point_cloud, p_center=center, l_axes=axes_lengths)

    # Visualize and save filtered point cloud
    o3d_visualizer(window_name="Filtered Point Cloud", geom1=filtered_cloud, save=False)
