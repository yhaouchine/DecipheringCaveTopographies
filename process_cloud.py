# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "novembre 2024"

import open3d as o3d
import numpy as np

""" Visualization with Open3D """

background_color = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
}


def o3d_visualizer(window_name, data_sets=None, data=None, multiple_data=False, color_name="white"):
    back_color = background_color.get(color_name.lower(), [1.0, 1.0, 1.0])
    open3d_visualizer = o3d.visualization.VisualizerWithEditing()
    open3d_visualizer.create_window(window_name=window_name, width=1280, height=800)
    open3d_visualizer.get_render_option().background_color = back_color

    # If multiple data is specified, let the user choose which geometries to display
    if multiple_data and data_sets:
        print("Select geometries to display:")
        for i, (label, geom) in enumerate(data_sets):
            print(f"{i}: {label} ")  # Display label and type of geometry
        geom_indices = input("Enter the indices of the geometries you want to display, separated by commas: ")
        geom_indices = [int(i.strip()) for i in geom_indices.split(',')]

        for i in geom_indices:
            if i < len(data_sets):
                open3d_visualizer.add_geometry(data_sets[i][1])  # Add the geometry from the tuple
            else:
                print(f"Index {i} is out of range.")
    elif data:
        open3d_visualizer.add_geometry(data)
    else:
        raise ValueError("No geometry provided. Please specify 'data' or 'data_sets'.")

    open3d_visualizer.run()
    open3d_visualizer.destroy_window()
    return open3d_visualizer


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

data_sets_list = []                                                         # initializing empty list of data
point_cloud_name = "cave_res_1cm.ply"
point_cloud = o3d.io.read_point_cloud("point_clouds/" + point_cloud_name)   # importing point cloud
data_sets_list.append((point_cloud_name, point_cloud))                      # adding point cloud to data list

# Creating first visualizer
visualizer = o3d_visualizer(window_name=point_cloud_name, data=point_cloud)

# Selecting points in the visualizer
selected_indices = visualizer.get_picked_points()
print("Selected points: ", selected_indices)

# Extracting the selected points
selected_points = point_cloud.select_by_index(selected_indices)
selected_points_coordinates = np.asarray(selected_points.points)
if not selected_indices:
    print("No points selected. Exiting.")
    exit()

# Position the cut on the point
cut_position = selected_points_coordinates[0][0]
thickness = 2

cross_section_pc = extract_cross_section(point_cloud, cut_position, thickness)
data_sets_list.append(("Cross-section points", cross_section_pc))

# Creating the second interactive visualizer for the cross-section
visualizer2 = o3d_visualizer(window_name=point_cloud_name + ": Cross-section", data=cross_section_pc)

# Selecting point in the cross-section from which the Bounding Box is created
center_indice = visualizer2.get_picked_points()
print("Selected points in cross_section: ", center_indice)

if not center_indice:
    print("No point selected in cross-section. Exiting.")
    exit()

# Getting the selected point
bbox_center = np.asarray(cross_section_pc.points)[center_indice[0]]
print("selected center of bounding box: ", bbox_center)

bbox_size = 1
bbox = create_bounding_box(bbox_center, bbox_size)
data_sets_list.append(("Bounding box", bbox))

visualizer3 = o3d_visualizer(window_name="PC + Bbox", data_sets=data_sets_list, multiple_data=True,
                             color_name="grey")

# Filter points outside the bounding box
filtered_pc = cross_section_pc.crop(bbox)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries(
    [filtered_pc],
    window_name="Filtered Point Cloud",
    width=1280,
    height=800
)
