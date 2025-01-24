# -*- coding:utf-8 -*-

import sys
import os
import open3d as o3d
import numpy as np
from process_cloud import o3d_visualizer


def create_ellipsoid_from_two_points(point_cloud, selected_indices, wireframe=False, resolution=30):
    """
    Creates an ellipsoid based on two selected points: one at the top and one at the bottom of the feature.
    @param point_cloud: The point cloud object.
    @param selected_indices: Indices of the two points selected by the user (top and bottom).
    @param wireframe: Whether to display the ellipsoid as wireframe.
    @param resolution: Resolution of the ellipsoid mesh.
    @return: Open3D geometry (TriangleMesh for solid or LineSet for wireframe).
    """
    if len(selected_indices) != 2:
        print("You need to select exactly two points.")
        sys.exit()

    # Get the coordinates of the selected points
    point_top = np.asarray(point_cloud.points)[selected_indices[0]]
    point_bottom = np.asarray(point_cloud.points)[selected_indices[1]]

    # Calculate the center of the ellipsoid (mean of the two selected points)
    center = (point_top + point_bottom) / 2

    # Calculate the vertical distance between the two points and define axes lengths
    vertical_distance = np.linalg.norm(point_top - point_bottom)
    axes_lengths = [1.0, 1.0, vertical_distance / 2]  # You can adjust these values to suit your needs

    # Create the ellipsoid using the calculated parameters
    ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    ellipsoid.scale(1.0, ellipsoid.get_center())  # Ensure unit sphere scaling
    ellipsoid.transform(np.diag([axes_lengths[0], axes_lengths[1], axes_lengths[2], 1.0]))  # Scale axes
    ellipsoid.translate(center)  # Translate to center

    if wireframe:
        # Convert the mesh to wireframe using edges
        edges = []
        for triangle in np.asarray(ellipsoid.triangles):
            edges.append([triangle[0], triangle[1]])
            edges.append([triangle[1], triangle[2]])
            edges.append([triangle[2], triangle[0]])

        edges = np.array(edges)
        unique_edges = np.unique(edges, axis=0)  # Remove duplicate edges
        lines = unique_edges.tolist()

        # Extract points from the mesh
        points = np.asarray(ellipsoid.vertices)
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([1, 0, 0])  # Red wireframe
        return line_set, center, axes_lengths
    else:
        return ellipsoid, center, axes_lengths


def filter_points_in_ellipsoid(point_cloud, center, axes_lengths):
    """
    Filters points within an ellipsoid defined by center and semi-axes.
    @param point_cloud: Open3D point cloud object.
    @param center: Center of the ellipsoid (x, y, z).
    @param axes_lengths: Lengths of the semi-axes (a, b, c).
    @return: Filtered point cloud.
    """
    points = np.asarray(point_cloud.points)
    relative_positions = points - center
    normalized_distances = (
            (relative_positions[:, 0] / axes_lengths[0]) ** 2
            + (relative_positions[:, 1] / axes_lengths[1]) ** 2
            + (relative_positions[:, 2] / axes_lengths[2]) ** 2
    )
    mask = normalized_distances > 1.0  # Points outside ellipsoid
    filtered_points = points[mask]
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
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
    filtered_cloud = filter_points_in_ellipsoid(point_cloud, center=center, axes_lengths=axes_lengths)

    # Visualize and save filtered point cloud
    o3d_visualizer(window_name="Filtered Point Cloud", geom1=filtered_cloud, save=False)
