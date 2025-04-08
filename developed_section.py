import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from typing import List
import time
from scipy.spatial import cKDTree
import logging

class DevelopedSection:
    def __init__(self):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.pc = None
        self.points_3d = None
        self.selected_idx = None
        self.selected_pts = None
        self.interpolated_line = None
        self.mean = None
        self.pca_axes = None
        self.points_2d = None
        self.projected_points = None
        self.developed_section_ini = None
        self.developed_section_projected = None


    def load_cloud(self, pc_name: str, parent_folder: str):
        """
        Loads a point cloud from a file.
        """
        pc_path = f"{parent_folder}/{pc_name}"
        pcd = o3d.io.read_point_cloud(pc_path)
        if pcd.is_empty():
            raise ValueError(f"Point cloud {pc_name} is empty or not found.")
        
        self.pc = pcd
        self.points_3d = np.asarray(self.pc.points)

    def select_line_of_cut(self) -> List[int]:
        """
        Allows manual selection of points to define the cutting line.
        Returns the indices of the selected points.
        """
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window("Select Cutting Line")
        vis.add_geometry(self.pc)
        vis.run()
        vis.destroy_window()
        self.selected_idx = vis.get_picked_points()
        self.selected_pts = self.points_3d[self.selected_idx]
        if not self.selected_idx:
            raise ValueError("No points were selected for the cutting line.")
        return self.selected_idx, self.selected_pts

    def sort_points(self):
        """
        Sort the selected points based on their Euclidean distance from the first point.
        """

        if self.selected_pts is None or len(self.selected_pts) < 2:
            raise ValueError("Selected points are not defined or contain less than two points.")
        
        # Calculate the Euclidean distances of all points from the first point
        dist = np.linalg.norm(self.selected_pts - self.selected_pts[0], axis=1)

        # Sort the points based on the distances
        sorted_indices = np.argsort(dist)
        self.selected_pts = self.selected_pts[sorted_indices]
        self.selected_idx = np.array(self.selected_idx)[sorted_indices]

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

    def extract_section(self, tolerance: float = 0.01) -> np.ndarray:
        """
        Fast extraction of points near the interpolated cutting line using a KDTree.
        """
        if self.pc is None or self.interpolated_line is None:
            raise ValueError("Point cloud or interpolated line is not defined.")

        points = np.asarray(self.pc.points)
        tree = cKDTree(points[:, :2])  # Only use X,Y for 2D proximity
        indices = set()

        for p in self.interpolated_line:
            nearby = tree.query_ball_point(p[:2], r=tolerance)
            indices.update(nearby)

        if not indices:
            raise ValueError("No points were extracted along the cutting line.")

        self.developed_section_ini = points[list(indices)]
        return self.developed_section_ini
    
    def pca_projection(self, diagnosis: bool = False, visualize: bool = False) -> np.ndarray:
        """
        Project the 3D points onto a 2D plane using Principal Component Analysis (PCA).
        
        Parameters:
        -----------
            - diagnosis: bool, optional (default=False)
                Whether to display the PCA diagnosis plot.
            - visualize: bool, optional (default=False)
                Whether to display the 2D projection of the point cloud.

        Returns:
        --------
        A NumPy array of shape (n, 2) containing the 2D coordinates of the projected points.
        """

        if self.developed_section_ini is None or len(self.developed_section_ini) == 0:
            raise ValueError("No points available in 'developed_section_ini' for PCA projection.")
        
        # Maintain Z axis vertical
        z_axis = np.array([0, 0, 1])

        self.mean = np.mean(self.developed_section_ini, axis=0)
        centered_points = self.developed_section_ini - self.mean

        pca = PCA(n_components=3)
        pca.fit(centered_points)
        self.pca_axes = pca.components_

        self.points_2d = pca.transform(centered_points)[:, :2]
        self.projected_points = self.points_2d

        if diagnosis:
            self._diagnose_pca(pca)

        if visualize:
            self._visualize_pca(pca)

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
        ax3d.scatter(self.developed_section_ini[:, 0], self.developed_section_ini[:, 1], self.developed_section_ini[:, 2], c='black', s=1, alpha=0.6,
                     label='3D Points')
        ax3d.set_title("3D Point Cloud")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        # Setting equal aspect ratio for 3D plot
        max_range = np.array([self.developed_section_ini[:, 0].max() - self.developed_section_ini[:, 0].min(),
                              self.developed_section_ini[:, 1].max() - self.developed_section_ini[:, 1].min(),
                              self.developed_section_ini[:, 2].max() - self.developed_section_ini[:, 2].min()]).max() / 2.0

        mid_x = (self.developed_section_ini[:, 0].max() + self.developed_section_ini[:, 0].min()) * 0.5
        mid_y = (self.developed_section_ini[:, 1].max() + self.developed_section_ini[:, 1].min()) * 0.5
        mid_z = (self.developed_section_ini[:, 2].max() + self.developed_section_ini[:, 2].min()) * 0.5

        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        # Adding arrows for the principal components
        scale = 0.1 * np.linalg.norm(self.developed_section_ini.max(axis=0) - self.developed_section_ini.min(axis=0))
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

    def display_section(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.projected_points[:, 0], self.projected_points[:, 1], c='blue', s=1, alpha=0.6, label='Developed Section Points')
        plt.title("Developed Section Visualization")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.axis('equal')
        plt.legend()
        plt.show()
        
if __name__ == "__main__":

    point_cloud_name = "cave_res_1cm.ply"
    save_folder = "point_clouds"

    try:
        # Load the point cloud
        start_time = time.perf_counter()
        developed_section_instance = DevelopedSection()
        developed_section_instance.load_cloud(pc_name=point_cloud_name, parent_folder=save_folder)
        print(f"Loading point cloud took {time.perf_counter() - start_time:.4f} seconds")
        
        # Select points for the cutting line
        developed_section_instance.select_line_of_cut()
        
        # Sort the selected points
        start_time = time.perf_counter()
        developed_section_instance.sort_points()
        print(f"Sorting points took {time.perf_counter() - start_time:.4f} seconds")

        # Interpolate the cutting line
        start_time = time.perf_counter()
        developed_section_instance.interpolate_line(auto_resolution=True, resolution=0.01)
        print(f"Interpolating cutting line took {time.perf_counter() - start_time:.4f} seconds")
        
        # Extract points along the cutting line
        start_time = time.perf_counter()
        developed_section_instance.extract_section(tolerance=0.02)
        print(f"Extracting points along the line took {time.perf_counter() - start_time:.4f} seconds")
        
        # Project the extracted points to 2D
        start_time = time.perf_counter()
        developed_section_instance.pca_projection(diagnosis=True, visualize=True)
        print(f"Projecting to 2D took {time.perf_counter() - start_time:.4f} seconds")

        developed_section_instance.display_section()
    
    except Exception as e:
        print(f"An error occurred: {e}")

#TODO:
# - Add sorting of the points in the developed section in order to avoid crossing lines
# - Ajust the height value on the graph
# - Add a function to save the developed section as a .ply file
# - Adapt the PCA projection for cases where the points are more distributed in the horizontal plane than in the vertical one