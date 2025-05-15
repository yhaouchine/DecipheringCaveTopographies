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
import logging
import os
from open3d.cpu.pybind.geometry import PointCloud, Geometry
from pathlib import Path
from tkinter import simpledialog, messagebox, Tk, filedialog
from typing import Tuple, Union, Optional
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


class Section:
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
        self.filename = None
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
        self.parent_folder = os.path.dirname(pc_path)
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

    def extract_nearby_points(self, tolerance: float = 0.01) -> np.ndarray:
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

    def display(self, points):
        """
        Display the extracted section using PyVista.
        """
        if points is None or len(points) == 0:
            raise ValueError("No points available to display.")

        plotter = pv.Plotter(window_size=[1280, 800])
        cloud = pv.PolyData(points)
        plotter.add_mesh(cloud, color='blue', point_size=3, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.set_background(self.background_color["white"])
        plotter.show_bounds(
            grid='back',
            location = 'outer',
            all_edges=False,
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=4,
            bold = True,
            font_family = 'arial',
            font_size = 10,
            )
        plotter.add_title("Extracted Section", font_size=18, font='arial')
        plotter.show(title="section Visualization")

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
        if ply_path:
            self.filename = ply_path.split("/")[-1]
        else:
            raise ValueError("Save operation was canceled. No file path provided.")

        self.filename = ply_path.split("/")[-1]
        o3d.io.write_point_cloud(ply_path, pc)

        # Save the interpolated line as an npy array if the instance is DevelopedSection and not PCASection
        if self.interpolated_line is None:
            raise ValueError("Interpolated line is not defined. Cannot save as .npy file.")
        npy_path = ply_path.replace(".ply", "_interpolated_line.npy")
        np.save(npy_path, self.interpolated_line)
        print(f"Interpolated line saved as: {npy_path}")
        
        messagebox.showinfo("Info", f"Section saved as: {self.filename}") if ply_path else print("Section not saved.")

class PCASection(Section):
    def __init__(self, pc = None, position = None, thickness = None):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        super().__init__(pc, position, thickness)
        self.pca_axes = None
        self.projected_points = None
        self.mean = None
        self.z_axis = np.array([0, 0, 1])
     
    def pca_projection(self, diagnosis: bool = False) -> np.ndarray:
        """
        Project the 3D points onto a 2D plane using Principal Component Analysis (PCA).
        
        Parameters:
        -----------
            - diagnosis: bool, optional (default=False)
                Whether to display the PCA diagnosis plot.

        Returns:
        --------
        A NumPy array of shape (n, 2) containing the 2D coordinates of the projected points.
        """

        if self.section is None or len(self.section) == 0:
            raise ValueError("No points available in 'section' for PCA projection.")
        
        start_time = time.perf_counter()
        self.mean = np.mean(self.section, axis=0)
        centered_points = self.section - self.mean

        pca = PCA(n_components=3)
        pca.fit(centered_points)
        self.pca_axes = pca.components_

        # Ensure the vertical axis is the one closest to Z
        vertical_idx = np.argmax(np.abs(np.dot(self.pca_axes, self.z_axis)))
        self.pca_axes[[1, vertical_idx]] = self.pca_axes[[vertical_idx, 1]]

        self.projected_points = pca.transform(centered_points)[:, :2]

        if diagnosis:
            self._diagnose_pca(pca)
            self._visualize_pca(pca)

        print(f"PCA performed successfuly in {time.perf_counter() - start_time:.4f} seconds")

        return self.projected_points

    def _diagnose_pca(self, pca: PCA):
        """
        Display a diagnosis of the Principal Component Analysis (PCA) results.
        
        Parameters:
        -----------
            - pca: PCA
                The PCA object fitted on the centered point cloud.
        """

        self.logger.info("==================== PCA Diagnosis ====================")

        # 1. Verify the eigenvalues as it indicates the variance of the data along the principal components
        eigenvalues = pca.explained_variance_
        self.logger.info("----- Eigenvalues -----")
        for i, (eig) in enumerate(eigenvalues, 1):
            self.logger.info(f"Eigenvalue {i}: {eig:.4f}")
        self.logger.info("")

        # 2. Verify the principal components
        self.logger.info("----- Principal components -----")
        for i, axis in enumerate(self.pca_axes, 1):
            self.logger.info(f"Axis PC{i}: {axis}")
        self.logger.info("")

        # 3. Verify orthogonality of the principal components
        dot_product = np.dot(self.pca_axes[0], self.pca_axes[1])
        self.logger.info(f"PC1 . PC2 = {dot_product:.4f}")

        self.logger.info("=================== END of Diagnosis ===================")

    def _visualize_pca(self, pca: PCA):
        """
        Display a 3D and 2D visualization of the point cloud and its principal components.

        Parameters:
        -----------
        pca: PCA
            The PCA object fitted on the centered point cloud.
        """

        fig = plt.figure(figsize=(12, 5))

        # 3D view
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.scatter(self.section[:, 0], self.section[:, 1], self.section[:, 2], c='black', s=1, alpha=0.6,
                     label='3D Points')
        ax3d.set_title("3D Point Cloud")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        # Setting equal aspect ratio for 3D plot
        max_range = np.array([self.section[:, 0].max() - self.section[:, 0].min(),
                              self.section[:, 1].max() - self.section[:, 1].min(),
                              self.section[:, 2].max() - self.section[:, 2].min()]).max() / 2.0

        mid_x = (self.section[:, 0].max() + self.section[:, 0].min()) * 0.5
        mid_y = (self.section[:, 1].max() + self.section[:, 1].min()) * 0.5
        mid_z = (self.section[:, 2].max() + self.section[:, 2].min()) * 0.5

        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        # Adding arrows for the principal components
        scale = 0.1 * np.linalg.norm(self.section.max(axis=0) - self.section.min(axis=0))
        colors = ['red', 'green', 'blue']
        labels = ['PC1', 'PC2', 'PC3']
        for i, (axis, color, label) in enumerate(zip(self.pca_axes, colors, labels), 1):
            ax3d.quiver(*self.mean, *axis * scale, color=color, label=label)

        ax3d.legend()

        # 2D view
        ax2d = fig.add_subplot(122)
        ax2d.scatter(self.projected_points[:, 0], self.projected_points[:, 1], c='black', s=1, alpha=0.6,
                     label='2D Projection')
        ax2d.set_title("2D Projection of the Point Cloud")
        ax2d.set_xlabel("PC1")
        ax2d.set_ylabel("PC2")
        ax2d.axis('equal')

        # Create vectors for the principal components
        pc1_3d = self.pca_axes[0] * 0.2
        pc2_3d = self.pca_axes[1] * 0.2

        # Transform the vectors to the 2D plane
        pc1_2d = pca.transform([pc1_3d])
        pc2_2d = pca.transform([pc2_3d])

        arrow_scale_2d = 20
        ax2d.arrow(0, 0, pc1_2d[0, 0] * arrow_scale_2d, pc1_2d[0, 1] * arrow_scale_2d, color='red', width=0.01,
                   head_width=0.4, label='PC1')
        ax2d.arrow(0, 0, pc2_2d[0, 0] * arrow_scale_2d, pc2_2d[0, 1] * arrow_scale_2d, color='green', width=0.01,
                   head_width=0.4, label='PC2')

        ax2d.legend()

        plt.tight_layout()
        plt.show()

    def compute(self, show: bool = True, diagnosis: bool = False):
        # Compute the PCA projection
        self.pca_projection(diagnosis=diagnosis)

        if show:
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)
            ax.scatter(self.projected_points[:, 0], self.projected_points[:, 1], c='black', s=1, alpha=0.6,
                        label='2D Projection')
            ax.set_title("2D Projection of the Point Cloud")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.axis('equal')
            ax.legend()
            plt.show()
        
        return self.projected_points

class DevelopedSection(Section):
    def __init__(self, pc: Optional[PointCloud] = None):
        super().__init__(pc=pc)
        self.developed_section = None
        self.interpolated_line = None

    def compute(self, show: bool = True):
        # compute cumulative distance along the densified line
        deltas = np.linalg.norm(np.diff(self.interpolated_line, axis=0), axis=1)
        cumdist = np.hstack([[0], np.cumsum(deltas)])
        # project each section point
        developed = []
        tree_line = cKDTree(self.interpolated_line[:, :2])
        dists, idxs = tree_line.query(self.section[:, :2])
        for pt, i in zip(self.section, idxs):
            developed.append([cumdist[i], pt[2]])  # X: distance, Y: Z
        self.developed_section = np.array(developed)

        if show:
            if self.developed_section is None:
                raise ValueError("Developed section not computed yet.")
            plt.figure(figsize=(12,6))
            plt.scatter(self.developed_section[:,0], self.developed_section[:,1], s=1)
            plt.xlabel('Developed distance (m)')
            plt.ylabel('Height Z (m)')
            plt.title('Developed Section')
            plt.axis('equal')
            plt.show()
        
        return self.developed_section

def extract_cross_section(tolerance: float) -> None:
    o3d.visualization.gui.Application.instance.initialize()

    pcp_instance = Section()
    pcp_instance.load_cloud()

    while True:
        pcp_instance.visualizer(window_name="Point Cloud", geom1=pcp_instance.pc, save=False)
        pcp_instance.sort_points()
        pcp_instance.interpolate_line(auto_resolution=True, resolution=0.005)
        pcp_instance.extract_nearby_points(tolerance=tolerance)
        pcp_instance.display(points=pcp_instance.section)

        user_input = messagebox.askyesnocancel("Save Section", "Do you want to save the section?")
        if user_input is True:
            pcp_instance.to_ply(points=pcp_instance.section)
        elif user_input is False:
            print("The section is not saved.")
        else:
            print("Process cancelled by the user.")

        another_section = messagebox.askyesno("Extract Another Section", "Do you want to extract another section?")
        if not another_section:
            print("Exiting the process.")
            break

if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    tolerance = 0.05
    extract_cross_section(tolerance=tolerance)

# TODO: 
# Add UI to select the method and tolerance.
# Fix the Save function.
