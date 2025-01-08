# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "novembre 2024"

import pyvista as pv
import open3d as o3d
import numpy as np

""" Visualization with pyvista """


def load_cloud(filepath):
    cloud = pv.read(filepath)
    return cloud


def visualize_cloud(pts):
    cloud = pv.PolyData(pts)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, point_size=5, render_points_as_spheres=False)
    plotter.set_background("grey", top="white")  # Background color
    plotter.enable_eye_dome_lighting()  # Eye Dome Lighting
    #plotter.render_window.SetMultiSamples(8)  # Anti-aliasing
    #plotter.enable_point_picking(callback=lambda p: print(f"Selected point : {p}"))
    plotter.show()


#point_cloud = load_cloud("point_clouds/cave_res_1cm_cross_section.ply")
#visualize_cloud(point_cloud)


""" Visualization with Open3D """

# initializing GUI instance
o3d.visualization.gui.Application.instance.initialize()

# Importing point cloud
point_cloud_name = "cave_res_1cm.ply"
point_cloud = o3d.io.read_point_cloud("point_clouds/" + point_cloud_name)

# Creating the visualizer
visualizer = o3d.visualization.VisualizerWithEditing()
visualizer.create_window(window_name=point_cloud_name, width=1280, height=800)
visualizer.add_geometry(point_cloud)
visualizer.run()
visualizer.destroy_window()

# Selecting points in the visualizer
selected_indices = visualizer.get_picked_points()
print("Selected points: ", selected_indices)

# Extracting the selected points
selected_points = point_cloud.select_by_index(selected_indices)
selected_points_coordinates = np.asarray(selected_points.points)
if len(selected_points_coordinates) == 0:
    print("No point selected")

else:
    # Position the cut on the x coordinate of the point
    cut_position = selected_points_coordinates[0][0]

    # Define the thickness of the cross-section
    thickness = 2

    # Extracting points within the cross-section thickness
    points = np.asarray(point_cloud.points)
    mask = (points[:, 0] > cut_position - thickness / 2) & (points[:, 0] < cut_position + thickness / 2)

    # Create a new point cloud with the extracted points
    cut_points = points[mask]
    cut_pcd = o3d.geometry.PointCloud()
    cut_pcd.points = o3d.utility.Vector3dVector(cut_points)

    # Display the new point cloud
    o3d.visualization.draw_geometries(
        [cut_pcd],
        window_name=point_cloud_name + ": Cross-section point cloud",
        width=1280,
        height=800
    )


