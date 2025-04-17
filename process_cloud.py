# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "November 2024"

import sys
import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
import pyvista as pv
from open3d.cpu.pybind.geometry import PointCloud, Geometry
from pathlib import Path
from tkinter import simpledialog, messagebox, Tk, filedialog
from typing import Tuple, Union, Optional
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

class CrossSection:
    def __init__(self, pc: Optional[PointCloud] = None, position: Optional[np.ndarray] = None,
                 thickness: Optional[float] = None):
        self.pc = pc
        self.pc_name = None
        self.parent_folder = None
        self.position = position
        self.thickness = thickness
        self.cross_section = None
        self.points_3d = None
        self.selected_idx = None
        self.selected_pts = None
        self.interpolated_line = None
        self.section = None
        self.pc_name = None
        self.background_color = {
            "white": [1.0, 1.0, 1.0],
            "grey": [0.5, 0.5, 0.5],
            "black": [0.0, 0.0, 0.0],
        }

    def load_cloud(self):
        """
        Load a point cloud from a file using Open3D point cloud reading function.        
        """
        start_time = time.perf_counter()
        pc_path = filedialog.askopenfilename(title="Select Point Cloud file", filetypes=[("Point Cloud files", "*.ply")])
        self.pc_name = pc_path.split("/")[-1].split(".")[0]
        self.pc = o3d.io.read_point_cloud(pc_path)
        if self.pc.is_empty():
            raise ValueError("The point cloud is empty or not loaded correctly.")
        self.points_3d = np.asarray(self.pc.points)
        print(f"Point cloud {pc_path} loaded successfuly in {time.perf_counter() - start_time:.4f} seconds")
        return self.pc

    def visualizer(self, window_name: str, geom1: Geometry = None, geom2: Geometry = None, save: Optional[bool] = None,
                   color_name: str = "white", filename: Union[Path, str, None] = None) -> Union[np.ndarray, None]:
        """
        Function to visualize the point cloud and select points in the visualizer.

        Parameters:
        -----------
            - window_name: str
                 Name of the window
            - geom1: open3d.geometry.Geometry
                 First geometry to visualize
            - geom2: open3d.geometry.Geometry
                 Second geometry to visualize (optional)
            - save: bool
                 If True, the point cloud is saved
            - color_name: str
                 Name of the color of the background
            - filename: str
                 Name of the file to save the point cloud
        
        Returns:
        --------
            - pick_point: np.ndarray
                 Array of the picked points
        """

        save_folder = Path("saved_clouds")
        save_folder.mkdir(parents=True, exist_ok=True)
        back_color = self.background_color.get(color_name.lower(), [1.0, 1.0, 1.0])

        # Handeling two geometries
        if geom1 and geom2:
            o3d.visualization.draw_geometries(
                [geom1, geom2],
                window_name=window_name,
                width=1280,
                height=800
            )
            return None

        # Use VisualizerWithEditing (to allow picking points)
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

        # Handeling the saving process
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
            elif user_choice is False:
                return None
            
        self.selected_idx = visualizer.get_picked_points()
        self.selected_pts = self.points_3d[self.selected_idx]
        if not self.selected_idx:
            raise ValueError("No points were selected for the cutting line.")
        
        visualizer.destroy_window()

        return self.selected_idx, self.selected_pts

    def sort_points(self):
        """
        Sort the selected points based on their Euclidean distance from the first point.
        """

        if self.selected_pts is None or len(self.selected_pts) < 2:
            raise ValueError("Selected points are not defined or contain less than two points.")
        
        start_time = time.perf_counter()
        
        # Calculate the Euclidean distances of all points from the first point
        dist = np.linalg.norm(self.selected_pts - self.selected_pts[0], axis=1)

        # Sort the points based on the distances
        sorted_indices = np.argsort(dist)
        self.selected_pts = self.selected_pts[sorted_indices]
        self.selected_idx = np.array(self.selected_idx)[sorted_indices]

        print(f"Points sorted successfuly in {time.perf_counter() - start_time:.4f} seconds")

    def interpolate_line(self, auto_resolution: bool = True, resolution: float = 0.01, nb_points: int = 500000) -> np.ndarray:
        """
        Interpolates a continuous line from selected points.

        Parameters:
        -----------
            - auto_resolution: bool, optional (default=True)
                If True, the number of points for interpolation is automatically determined based on the length of the line and the resolution.
                If False, the number of points is fixed and determined by the `nb_points` parameter.
            - resolution: float, optional (default=0.01)
                The desired spacing between points along the interpolated line when `auto_resolution` is True.
                This value is used to calculate the number of points based on the line's length.
        """
        if self.selected_idx is None or len(self.selected_idx) < 2:
            raise ValueError("At least two points are required for interpolation.")
        
        start_time = time.perf_counter()

        t = np.linspace(0, 1, len(self.selected_pts))
        interp_func = interp1d(t, self.selected_pts, axis=0, kind='linear')

        if auto_resolution:
            # Automatically determine the number of points based on the length of the line
            if self.selected_pts is None or len(self.selected_pts) < 2:
                raise ValueError("Selected points are not defined or contain fewer than two points.")
            
            length = np.linalg.norm(self.selected_pts[-1] - self.selected_pts[0])
            nb_points = int(length / resolution) + 1

        # Perform the interpolation using the calculated or provided number of points
        self.interpolated_line = interp_func(np.linspace(0, 1, nb_points))

        print(f"Line of cut interpolated successfuly in {time.perf_counter() - start_time:.4f} seconds")

    def extract_section(self, tolerance: float = 0.01) -> np.ndarray:
        """
        Fast extraction of points near the interpolated cutting line using a KDTree.
        """
        if self.pc is None or self.interpolated_line is None:
            raise ValueError("Point cloud or interpolated line is not defined.")

        start_time = time.perf_counter()

        points = np.asarray(self.pc.points)
        tree = cKDTree(points[:, :2])  # Only use X,Y for 2D proximity
        indices = set()

        for p in self.interpolated_line:
            nearby = tree.query_ball_point(p[:2], r=tolerance)
            indices.update(nearby)

        if not indices:
            raise ValueError("No points were extracted along the cutting line.")

        self.section = points[list(indices)]
        
        print(f"Section's points extracted successfuly in {time.perf_counter() - start_time:.4f} seconds")

        return self.section

    def display_section(self, points):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=1, alpha=0.6, label='Cross-section Points')
        ax.set_title("Cross-section Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axis("equal")
        ax.legend()
        plt.show()
    
    def display_pyvista(self, points):
        """
        Display the extracted section using PyVista.
        """
        if points is None or len(points) == 0:
            raise ValueError("No points available to display.")

        plotter = pv.Plotter()
        cloud = pv.PolyData(points)
        plotter.add_mesh(cloud, color='black', point_size=5, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.show(title="Cross-section Visualization")

    def to_ply(self, points):
        """
        Save the projected developed section to a PLY file.
        """

        if points is None or len(points) == 0:
            raise ValueError("No points available to save.")
        
        # Create a point cloud object
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        # Save the point cloud to a PLY file
        ply_path = filedialog.asksaveasfilename(defaultextension=".ply", confirmoverwrite=True, filetypes=[("PLY files", "*.ply")])
        filename = ply_path.split("/")[-1]
        o3d.io.write_point_cloud(ply_path, pc)
        messagebox.showinfo("Info", f"Section saved as: {filename}") if ply_path else print("Section not saved.")

    def extract(self, tolerance: float) -> PointCloud:
        """
        Function to extract a cross-section of the point cloud located at the picked point in the visualizer.
        If multiple points are selected in the visualizer, extract as many cross-sections.
        
        Parameters:
        -----------
        thickness: float
            Thickness of the cross-section
            
        Returns:
        --------
        cross_section: open3d.geometry.PointCloud
            Point cloud of the cross-section
        """

        selected_indices = self.visualizer(window_name="Select cut position", geom1=self.pc, save=False)
        if not selected_indices:
            sys.exit("No points selected.")

        selected_points = self.pc.select_by_index(selected_indices)
        selected_points_coordinates = np.asarray(selected_points.points)
        cut_positions = selected_points_coordinates[0][0]

        points = np.asarray(self.pc.points)
        mask = (points[:, 0] > cut_positions - tolerance / 2) & (points[:, 0] < cut_positions + tolerance / 2)
        cut_points = points[mask]
        cut_point_cloud = o3d.geometry.PointCloud()
        cut_point_cloud.points = o3d.utility.Vector3dVector(cut_points)
        self.cross_section = cut_point_cloud

        return self.cross_section

def extract_cross_section(tolerance: float):
    o3d.visualization.gui.Application.instance.initialize()
    pcp_instance = CrossSection()
    pcp_instance.load_cloud()
    pcp_instance.visualizer(window_name="Point Cloud", geom1=pcp_instance.pc, save=False)
    pcp_instance.sort_points()
    pcp_instance.interpolate_line(auto_resolution=True, resolution=0.005)
    pcp_instance.extract_section(tolerance=tolerance)
    #pcp_instance.display_section(pcp_instance.section)
    pcp_instance.display_pyvista(pcp_instance.section)
    user_input = messagebox.askyesnocancel("Save Section", "Do you want to save the section?")
    if user_input is True:
        pcp_instance.to_ply(points=pcp_instance.section)
    elif user_input is False:
        print("The section is not saved.")
    else:
        print("Process cancelled by the user.")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    tolerance = 0.02
    extract_cross_section(tolerance=tolerance)
