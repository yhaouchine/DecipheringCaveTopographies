# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "novembre 2024"

import sys
import open3d as o3d
import numpy as np

""" Visualization with Open3D """

background_color = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
}


def o3d_visualizer(window_name, geom1=None, geom2=None, color_name="white"):
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
    if geom1:
        visualizer.add_geometry(geom1)
    visualizer.run()
    picked_points = visualizer.get_picked_points()
    visualizer.destroy_window()
    return picked_points


def extract_cross_section(pc, position, e):
    points = np.asarray(pc.points)
    mask = (points[:, 0] > position - e / 2) & (points[:, 0] < position + e / 2)
    cut_points = points[mask]
    cut_pc = o3d.geometry.PointCloud()
    cut_pc.points = o3d.utility.Vector3dVector(cut_points)
    return cut_pc


def create_bounding_box(center_point, size):
    min_bound = center_point - size / 2
    max_bound = center_point + size / 2
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)


# initializing GUI instance
o3d.visualization.gui.Application.instance.initialize()

point_cloud_name = "cave_res_1cm.ply"
point_cloud = o3d.io.read_point_cloud("point_clouds/" + point_cloud_name)  # importing point cloud

# Selecting points in the visualizer
selected_indices = o3d_visualizer(window_name=point_cloud_name, geom1=point_cloud)
if not selected_indices:
    print("No points selected. Exiting.")
    sys.exit()

# Extracting the selected points
selected_points = point_cloud.select_by_index(selected_indices)
selected_points_coordinates = np.asarray(selected_points.points)

# Position the cut on the point
cut_position = selected_points_coordinates[0][0]
thickness = 2
cross_section_pc = extract_cross_section(point_cloud, cut_position, thickness)

# Selecting point in the cross-section from which the Bounding Box is created
center_index = o3d_visualizer(window_name=point_cloud_name + ": Cross-section", geom1=cross_section_pc)
if not center_index:
    print("No point selected in cross-section. Exiting.")
    sys.exit()

bbox_center = np.asarray(cross_section_pc.points)[center_index[0]]
bbox_size = 1
bbox = create_bounding_box(bbox_center, bbox_size)
bbox.color = [1.0, 0.0, 0.0]

visualizer3 = o3d_visualizer(window_name="PC + Bbox", geom1=cross_section_pc, geom2=bbox, color_name="grey")

# Filter points outside the bounding box
filtered_pc = cross_section_pc.crop(bbox)

# Visualize the filtered point cloud
visualizer4 = o3d_visualizer(window_name="Filtered PC", geom1=filtered_pc, color_name="grey")
