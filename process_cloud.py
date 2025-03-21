# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "November 2024"

import sys
import open3d as o3d
import numpy as np
from open3d.cpu.pybind.geometry import PointCloud, Geometry
from pathlib import Path
from tkinter import simpledialog, messagebox, Tk
from typing import Tuple, Union, Optional

background_color = {
    "white": [1.0, 1.0, 1.0],
    "grey": [0.5, 0.5, 0.5],
    "black": [0.0, 0.0, 0.0],
}

class PointCloudProcessor:
    def __init__(self, pc: Optional[PointCloud] = None, position: Optional[np.ndarray] = None, thickness: Optional[float] = None):
        """
        Constructor of the PointCloudProcessor class.
        """

        self.pc = pc
        self.pc_name = None
        self.parent_folder = None
        self.position = position
        self.thickness = thickness
        self.pick_point = None
        self.cross_section = None
        self.points_3d = None


    def load_cloud(self, pc_name: str, parent_folder: str):
        """
        Load a point cloud from a file using Open3D point cloud reading function.        
        """

        self.pc_name = pc_name
        self.parent_folder = parent_folder
        self.pc = o3d.io.read_point_cloud(f"{self.parent_folder}/{self.pc_name}")
        self.points_3d = np.asarray(self.pc.points)

        return self.pc

    def visualizer(self, window_name: str, geom1: Geometry = None, geom2: Geometry = None, save: Optional[bool] = None,
                   color_name: str = "white", filename: Union[Path, str,  None] = None) -> Union[np.ndarray, None]:
        """
        Function to visualize the point cloud and select points in the visualizer.

        Parameters:
        -----------
        window_name: str
            Name of the window
        geom1: open3d.geometry.Geometry
            First geometry to visualize
        geom2: open3d.geometry.Geometry
            Second geometry to visualize (optional)
        save: bool
            If True, the point cloud is saved
        color_name: str
            Name of the color of the background
        filename: str
            Name of the file to save the point cloud
        
        Returns:
        --------
        pick_point: np.ndarray
            Array of the picked points
        """
        
        save_folder = Path("saved_clouds")
        save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        back_color = background_color.get(color_name.lower(), [1.0, 1.0, 1.0])

        if geom1 and geom2:
            o3d.visualization.draw_geometries(
                [geom1, geom2],
                window_name = window_name,
                width = 1280,
                height = 800
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

        # Save the point cloud if asked
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
        self.pick_point = visualizer.get_picked_points()
        visualizer.destroy_window()

        return self.pick_point
    
    def extract_cross_section(self, thickness: float) -> PointCloud:
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
        mask = (points[:, 0] > cut_positions - thickness / 2) & (points[:, 0] < cut_positions + thickness / 2)
        cut_points = points[mask]
        cut_point_cloud = o3d.geometry.PointCloud()
        cut_point_cloud.points = o3d.utility.Vector3dVector(cut_points)
        self.cross_section = cut_point_cloud

        return self.cross_section


if __name__ == "__main__":

    # initializing GUI instance
    o3d.visualization.gui.Application.instance.initialize()

    # Importing the point cloud
    point_cloud_name = "cave_res_1cm.ply"
    parent_folder = "point_clouds"

    pcp_instance = PointCloudProcessor()
    point_cloud = pcp_instance.load_cloud(pc_name=point_cloud_name, parent_folder=parent_folder)

    # Extracting the cross-section
    thickness = 0.1
    cross_section = pcp_instance.extract_cross_section(thickness)

    # Visualizing the cross-section
    pcp_instance.visualizer(window_name="Cross-Section", geom1=cross_section, save=None)