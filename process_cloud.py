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
    def __init__(self, pc: PointCloud, position: Optional[np.ndarray] = None, thickness: Optional[float] = None):
        """
        Constructor of the CrossSection class.
        """

        self.pc = pc
        self.position = position
        self.thickness = thickness
        self.pick_point = None
        self.cross_section = None
    
    def visualizer(self, window_name: str, geom: Geometry = None, save: Optional[bool] = None,
                   color_name: str = "white", filename: Union[Path, str,  None] = None) -> Union[np.ndarray, None]:
        """
        Function to visualize the point cloud and select points in the visualizer.

        Parameters:
        ----------
        window_name: str
            Name of the window
        geom: open3d.geometry.Geometry
            Geometry to visualize
        save: bool
            If True, the point cloud is saved
        color_name: str
            Name of the color of the background
        filename: str
            Name of the file to save the point cloud
        
        Returns:
        -------
        pick_point: np.ndarray
            Array of the picked points
        """
        
        save_folder = Path("saved_clouds")
        save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist
        back_color = background_color.get(color_name.lower(), [1.0, 1.0, 1.0])

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

        if geom:
            visualizer.add_geometry(geom)
        visualizer.run()

        root = Tk()
        root.withdraw()

        # Save the point cloud if asked
        if geom and save:
            if not filename:
                filename = simpledialog.askstring("File Name", "Enter a name for the point cloud (e.g., 'cloud.ply'):")
                save_path = save_folder / filename
                o3d.io.write_point_cloud(str(save_path), geom)
                print(f"Point cloud saved as: {save_path}")
        elif geom and save is None:
            user_choice = messagebox.askyesnocancel("Save Point Cloud", "Do you want to save the point cloud?")
            if user_choice is True:
                filename = simpledialog.askstring("File Name", "Enter a name for the point cloud (e.g., 'cloud.ply'):")
                save_path = save_folder / filename
                o3d.io.write_point_cloud(str(save_path), geom)
                print(f"Point cloud saved as: {save_path}")
            elif user_choice is False:
                return None
        self.pick_point = visualizer.get_picked_points()
        visualizer.destroy_window()

        return self.pick_point
    
    def extract_cross_section(self, cut_position: np.ndarray, thickness: float) -> PointCloud:
        """
        Function to extract a cross-section of the point cloud located at the picked point in the visualizer.
        If multiple points are selected in the vizualizer, extract as manu cross-sections.
        
        Parameters:
        ----------
        cut_position: np.ndarray
            Position of the cross-section

        thickness: float
            Thickness of the cross-section
            
        Returns:
        -------
        cross_section: open3d.geometry.PointCloud
            Point cloud of the cross-section
        """

        points = np.asarray(self.pc.points)
        mask = (points[:, 0] > cut_position - thickness / 2) & (points[:, 0] < cut_position + thickness / 2)
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
    point_cloud = o3d.io.read_point_cloud("point_clouds/" + point_cloud_name)

    # Creating the cross-section object
    cross_section = PointCloudProcessor(pc=point_cloud)

    # Selecting the cut position in the visualizer
    selected_indices = cross_section.visualizer(window_name=point_cloud_name, geom=point_cloud, save=False)
    print(selected_indices)
    print(len(selected_indices))
    if not selected_indices:
        sys.exit("No points selected.")
    

    selected_point = point_cloud.select_by_index(selected_indices)
    selected_point_coordinates = np.asarray(selected_point.points)
    cut_position = selected_point_coordinates[0][0]

    # Extracting the cross-section
    thickness = 0.1
    cross_section.extract_cross_section(cut_position, thickness)

    # Visualizing the cross-section
    cross_section.visualizer(window_name="Cross-Section", geom=cross_section.cross_section, save=None)