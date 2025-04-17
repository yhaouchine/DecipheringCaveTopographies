# -*- coding:utf-8 -*-

import open3d as o3d
import numpy as np
import tkinter as tk
from open3d.cpu.pybind.geometry import PointCloud
from process_cloud import CrossSection
from tkinter import simpledialog, messagebox, Tk, filedialog
from typing import Tuple, Optional, List


class Ellipsoid(object):
    def __init__(self, pc: PointCloud, resolution: int = 15, selected_i: Optional[List[int]] = None, selected_coords: Optional[List[float]] = None):
        self.resolution = resolution
        self.pc = pc
        self.filtered_pc = None
        self.dominant_axis = None
        self.selected_i = selected_i
        self.selected_coords = [np.array(c) for c in selected_coords] if selected_coords is not None else None
        self.center_point = None
        self.line_set = None
        self.semi_axes = None
        self.axis_labels = ['a_axis', 'b_axis', 'c_axis']

    def ellipsoid_from_2_points(self):
        p1_coords = np.array(self.selected_coords[0])
        p2_coords = np.array(self.selected_coords[1])
        self.center_point = np.mean([p1_coords, p2_coords], axis=0)
        distance_vector = np.abs(p2_coords - p1_coords)
        self.dominant_axis = np.argmax(distance_vector)
        self.semi_axes = np.ones(3)
        self.semi_axes[self.dominant_axis] = distance_vector[self.dominant_axis] / 2
        print(f"Dominant axis: {self.axis_labels[self.dominant_axis]} = {self.semi_axes[self.dominant_axis]}")
        return self.generate_ellipsoid_mesh()
    
    def ellipsoid_from_1_point(self):
        self.semi_axes = np.ones(3)
        self.center_point = np.asarray(self.pc.points)[self.selected_i[0]]
        return self.generate_ellipsoid_mesh()

    def generate_ellipsoid_mesh(self):
        meshed_ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=self.resolution)
        scaling_matrix = np.diag([self.semi_axes[0], self.semi_axes[1], self.semi_axes[2], 1.0])
        meshed_ellipsoid.transform(scaling_matrix)
        meshed_ellipsoid.translate(self.center_point)
        edges = np.array([[triangle[i], triangle[(i + 1) % 3]] for triangle in np.asarray(meshed_ellipsoid.triangles) for i in range(3)])
        unique_edges = np.unique(edges, axis=0)  # Remove duplicate edges
        lines = unique_edges.tolist()

        # Create a LineSet object to display the ellipsoid as a wireframe
        points = np.asarray(meshed_ellipsoid.vertices)
        self.line_set = o3d.geometry.LineSet()
        self.line_set.points = o3d.utility.Vector3dVector(points)
        self.line_set.lines = o3d.utility.Vector2iVector(lines)
        self.line_set.paint_uniform_color([1, 0, 0])  # Red color for the wireframe

        return self.line_set, self.center_point, self.semi_axes

    def update_ellipsoid(self):
        def on_submit():
            try:
                for i in range(3):
                    val = entries[i].get()
                    if val.strip():
                        new_val = float(val)
                        self.semi_axes[i] = new_val
            except ValueError:
                print("invalide input. Keeping previous values.")
            top.destroy()
            
        top = tk.Toplevel()
        top.title("Update the ellipsoid axes")

        entries = []

        for i, name in enumerate(self.axis_labels):
            tk.Label(top, text=f"{name.replace('_', '-')} :").grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(top)
            entry.insert(0, str(self.semi_axes[i]))
            entry.grid(row=i, column=1, padx=5, pady=5)
            entries.append(entry)

        submit_btn = tk.Button(top, text="Valider", command=on_submit)
        submit_btn.grid(row=3, column=0, columnspan=2, pady=10)

        top.transient()  # pour rester au-dessus de la fenêtre principale
        top.grab_set()
        top.wait_window()

        print(f"Updated ellipsoid with new semi-axes: "
                f"a={self.semi_axes[0]}, b={self.semi_axes[1]}, c={self.semi_axes[2]}.")

        return self.generate_ellipsoid_mesh()

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
            (relative_position[:, 0] / self.semi_axes[0]) ** 2
            + (relative_position[:, 1] / self.semi_axes[1]) ** 2
            + (relative_position[:, 2] / self.semi_axes[2]) ** 2
        )
        mask = normalized_distance > 1.0
        filtered_points = points[mask]
        self.filtered_pc = o3d.geometry.PointCloud()
        self.filtered_pc.points = o3d.utility.Vector3dVector(filtered_points)
        return self.filtered_pc

def clean_cloud():
    pcp_instance = CrossSection()
    point_cloud = pcp_instance.load_cloud()

    while True:
        print("Please select one or two points: ")
        try:
            selected_indices, _ = pcp_instance.visualizer(window_name=pcp_instance.pc_name, geom1=point_cloud, save=False)
        except ValueError as e:
            print("No points selected. The cloud is considered clean.")
            break
        selected_coords = [np.asarray(point_cloud.points)[i] for i in selected_indices]

        if not selected_indices:
            print("No points selected. The cloud is considered clean.")
            break

        ellipsoid_instance = Ellipsoid(pc=point_cloud, selected_i=selected_indices, selected_coords=selected_coords)

        if len(selected_indices) == 2:
            ellipsoid_instance.line_set, ellipsoid_instance.center_point, ellipsoid_instance.semi_axes = ellipsoid_instance.ellipsoid_from_2_points()
        
        elif len(selected_indices) == 1:
            ellipsoid_instance.line_set, ellipsoid_instance.center_point, ellipsoid_instance.semi_axes = ellipsoid_instance.ellipsoid_from_1_point()
        
        else:
            raise ValueError("Number of selected points must be 1 or 2.")

        while True:
            root = Tk()
            root.withdraw()

            pcp_instance.visualizer(window_name="Point Cloud with Ellipsoid", geom1=point_cloud, geom2=ellipsoid_instance.line_set)
            user_input = messagebox.askyesnocancel("Validation", "Does the ellipsoid fit the points?")

            if user_input is True:
                print("Ellipsoid confirmed. Removing points within...")
                break

            elif user_input is False:
                ellipsoid_instance.line_set, ellipsoid_instance.center_point, ellipsoid_instance.semi_axes = ellipsoid_instance.update_ellipsoid()

            elif user_input is None:
                print("Operation canceled")
                exit()

        filtered_cloud = ellipsoid_instance.filter_points()
        point_cloud = filtered_cloud

    user_input_2 = messagebox.askyesnocancel("Save", "Do you want to save the cloud?")
    
    if user_input_2 is True:
        ply_path = filedialog.asksaveasfilename(defaultextension=".ply", confirmoverwrite=True, filetypes=[("PLY files", "*.ply")])
        filename = ply_path.split("/")[-1]
        o3d.io.write_point_cloud(ply_path, point_cloud)
        print("Filtered point cloud saved as " + filename)

    elif user_input_2 is False:
        print("Process complete.")

    elif user_input_2 is None:
        print("Operation canceled")
        exit()

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    clean_cloud()