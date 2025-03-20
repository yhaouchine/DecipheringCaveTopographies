# -*- coding:utf-8 -*-

import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud
from process_cloud import PointCloudProcessor
from tkinter import simpledialog, messagebox, Tk
from typing import Tuple, Optional, List


class Ellipsoid(object):
    def __init__(self, pc: PointCloud, selected_i: Optional[List[int]] = None, resolution: int = 15):
        """
        Contructor of the Ellipsoid class.
        """

        self.resolution = resolution
        self.pc = pc
        self.filtered_pc = None
        self.a_axis = 1.0
        self.b_axis = 1.0
        self.c_axis = 1.0
        self.selected_i = selected_i
        self.center_point = None
        self.line_set = None
        self.l_axes = None
    
    def create(self) -> Tuple[o3d.geometry.LineSet, np.ndarray, list[float]]:
        """
        Function to create a wireframed ellipsoid, either by considering the c-axis (vertical axis) 
        of the ellipsoid as defined between two points selected bu the user. Or by defining the center of the
        ellispoid as the unique point selected by the user, and the length of the axes as determined bu the user.

        Returns:
        --------
        line_set: open3d.geometry.LineSet
            The wireframed ellispoid.

        center_point: np.ndarray
            The center of the ellipsoid.

        l_axes: list[float] 
            A list containing the lengths of the ellipsoid axes.
        """
        if self.selected_i is None:
            raise ValueError("selected_i cannot be None.")
        elif len(self.selected_i) == 2:
            # Get the coordinates of the selected points
            z_top = np.asarray(self.pc.points)[self.selected_i[0]]
            z_bottom = np.asarray(self.pc.points)[self.selected_i[1]]

            # Calculate the center of the ellipsoid (mean of the two selected points)
            self.center_point = (z_top + z_bottom) / 2
            # Calculate the vertical distance between the two points and define initial axes lengths
            self.c_axis = np.linalg.norm(z_top - z_bottom)
        elif len(self.selected_i) == 1:
            self.center_point = np.asarray(self.pc.points)[self.selected_i[0]]
        else:
            raise ValueError("The number of selected points must be 1 or 2.")

        # Define axes lengths
        self.l_axes = [self.a_axis / 2, self.b_axis / 2, self.c_axis / 2]  # Semi-axes lengths

        # Create a unit sphere and transform it into an ellipsoid
        meshed_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=self.resolution)      # Create a unit sphere
        meshed_ellipsoid.scale(1.0, meshed_ellipsoid.get_center())                                              # Ensure unit sphere scaling and center
        scaling_matrix = np.diag([self.l_axes[0], self.l_axes[1], self.l_axes[2], 1.0])                         # Create the scaling matrix to transform the ellipsoid
        meshed_ellipsoid.transform(scaling_matrix)                                                              # Transform the ellipsoid with the scaling matrix
        meshed_ellipsoid.translate(self.center_point)                                                           # Translate the ellipsoid to center point calculated
        
        # Convert the mesh to wireframe using edges
        edges = np.array([[triangle[i], triangle[(i + 1) % 3]] for triangle in np.asarray(meshed_ellipsoid.triangles) for i in range(3)])
        unique_edges = np.unique(edges, axis=0)  # Remove duplicate edges
        lines = unique_edges.tolist()

        # Create a LineSet object to display the ellipsoid as a wireframe
        points = np.asarray(meshed_ellipsoid.vertices)
        self.line_set = o3d.geometry.LineSet()
        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.paint_uniform_color([1, 0, 0])  # Red color for the wireframe

        return self.line_set, self.center_point, self.l_axes
    
    def filter_points(self) -> PointCloud:
        """
        Function to filter out the points located withing the ellipsoid.

        Returns:
        --------
        filtered_pc: PointCloud
            The cleaned point cloud.
        """
        points = np.asarray(self.pc.points)
        relative_position = points - self.center_point
        normalized_distance = (
            (relative_position[:, 0] / self.l_axes[0]) ** 2
            + (relative_position[:, 1] / self.l_axes[1]) ** 2
            + (relative_position[:, 2] / self.l_axes[2]) ** 2
        )
        mask = normalized_distance > 1.0
        filtered_points = points[mask]
        self.filtered_pc = o3d.geometry.PointCloud()
        self.filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
        return self.filtered_pc


if __name__ == "__main__":

    # Importing the point cloud
    point_cloud_name = "cross_section_3_45d_clean.ply"
    parent_folder = "saved_clouds/"
    point_cloud = o3d.io.read_point_cloud(parent_folder + point_cloud_name)

    while True:
        # Select points
        print("Please select one or two points: ")
        pcp_instance = PointCloudProcessor(pc=point_cloud)
        selected_indices = pcp_instance.visualizer(window_name=point_cloud_name, geom1=point_cloud, save=False)

        if not selected_indices:
            print("No points selected. The cloud is considered clean.")
            break

        # Create ellipsoid based on the two points selected
        ellipsoid_instance = Ellipsoid(pc=point_cloud, selected_i=selected_indices)
        ellipsoid_instance.line_set, ellipsoid_instance.center_point, ellipsoid_instance.l_axes = ellipsoid_instance.create()

        while True:
            # Visualize the point cloud with the ellipsoid
            pcp_instance.visualizer(window_name="Point Cloud with Ellipsoid", geom1=point_cloud, geom2=ellipsoid_instance.line_set)

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

                    ellipsoid_instance.a_axis = new_a_axis
                    ellipsoid_instance.b_axis = new_b_axis
                    ellipsoid_instance.c_axis = new_c_axis
                    ellipsoid_instance.line_set, ellipsoid_instance.center_point, ellipsoid_instance.l_axes = ellipsoid_instance.create()

                except Exception as e:
                    print(f"An error occurred: {e}. Please enter valid numeric values.")
            elif user_input is None:
                print("Operation canceled")
                exit()

        # Filter points inside the ellipsoid
        filtered_cloud = ellipsoid_instance.filter_points()
        point_cloud = filtered_cloud

    user_input_2 = messagebox.askyesnocancel("Save", "Do you want to save the cloud?")
    if user_input_2 is True:
        saved_point_cloud_name = simpledialog.askstring("Filename",
                                                        "Name of the file with extension (e.g. Cloud.ply): ")
        print("Saving filtered point cloud...")
        o3d.io.write_point_cloud("saved_clouds/" + saved_point_cloud_name, point_cloud)
        print("Filtered point cloud saved as " + saved_point_cloud_name + "'.")
    elif user_input_2 is False:
        print("Process complete.")
    elif user_input_2 is None:
        print("Operation canceled")
        exit()
