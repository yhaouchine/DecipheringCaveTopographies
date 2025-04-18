import numpy as np
import alphashape
import logging
import matplotlib.pyplot as plt
import open3d as o3d
import tkinter as tk
import time
import os
import vtk
from tkinter import filedialog, Tk, messagebox, ttk
from typing import Tuple, Optional
from open3d.cpu.pybind.geometry import PointCloud
from concave_hull import concave_hull
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, MultiPolygon

logger = logging.getLogger(__name__)


class ContourExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.voxel_size = None
        self.pc_name = None
        self.parent_folder = None
        self.points_3d = None
        self.points_2d = None
        self.contour = None
        self.mean = None
        self.pca_axes = None
        self.projected_points = None
        self.original_cloud: Optional[PointCloud] = None
        self.reduced_cloud: Optional[PointCloud] = None
        self.durations: Optional[float] = None
        self.area: Optional[float] = None
        self.perimeter: Optional[float] = None
        self.roughness: Optional[float] = None
        self.curvature: Optional[np.ndarray] = None

    def load_cloud(self):
        """
        Load a point cloud from a file using Open3D point cloud reading function.        
        """

        try:
            pc_path = filedialog.askopenfilename(title="Select Point Cloud file", filetypes=[("Point Cloud files", "*.ply")])
            self.pc_name = pc_path.split("/")[-1].split(".")[0]
            logger.info("")
            start_time = time.perf_counter()
            logger.info(f"===== Loading point cloud {self.pc_name}... =====")
            self.original_cloud = o3d.io.read_point_cloud(pc_path)
            self.points_3d = np.asarray(self.original_cloud.points)
            print(f"Point cloud {pc_path} loaded successfuly in {time.perf_counter() - start_time:.4f} seconds")
        except Exception as e:
            logger.error(f"An error occurred while loading the point cloud: {e}")
            raise

    def downsample(self, voxel_size: float):
        """
        Downsample the point cloud using voxel grid downsampling. 

        The 3D space containing the point cloud is divided into a grid of equally sized voxels. 
        The size of the voxels is determined by the voxel_size parameter.
        
        Each point in the point cloud is assigned to a voxel based on its coordinates. This is 
        done by determining which voxel the point falls into.
        
        For each voxel that contains one or more points, a single representative point is chosen by compute 
        the average (mean) position of all points in the voxel and use it as the representative point.

        Parameters:
        -----------
            - voxel_size: float
                The size of the voxel grid used for downsampling. The unit 
                correspond to the unit of the point cloud coordinates. 
        """

        try:
            self.reduced_cloud = self.original_cloud.voxel_down_sample(voxel_size=voxel_size)
            self.points_3d = np.asarray(self.reduced_cloud.points)
            if self.points_3d.shape[0] < 3:
                logger.warning("Not enough points to generate a contour.")
                raise ValueError("Not enough points to generate a contour.")
            logger.info("")
            logger.info("===== Downsampling the point cloud... =====")
            logger.info("Point cloud downsampled successfully.")

        except Exception as e:
            logger.error(f"An error occurred while downsampling the point cloud: {e}")
            raise

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

        self.mean = np.mean(self.points_3d, axis=0)
        centered_points = self.points_3d - self.mean
        covariance_matrix = np.cov(centered_points, rowvar=False)

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
        Display useful information on the Principal Component Analysis (PCA) results.
        
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
        Display a 3D and 2D-projected visualization of the point cloud and its principal components.

        Parameters:
        -----------
        pca: PCA
            The PCA object fitted on the centered point cloud.
        """

        fig = plt.figure(figsize=(12, 5))

        # 3D view
        ax3d = fig.add_subplot(121, projection='3d')
        ax3d.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='black', s=1, alpha=0.6,
                     label='3D Points')
        ax3d.set_title("3D Point Cloud")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")

        # Setting equal aspect ratio for 3D plot
        max_range = np.array([self.points_3d[:, 0].max() - self.points_3d[:, 0].min(),
                              self.points_3d[:, 1].max() - self.points_3d[:, 1].min(),
                              self.points_3d[:, 2].max() - self.points_3d[:, 2].min()]).max() / 2.0

        mid_x = (self.points_3d[:, 0].max() + self.points_3d[:, 0].min()) * 0.5
        mid_y = (self.points_3d[:, 1].max() + self.points_3d[:, 1].min()) * 0.5
        mid_z = (self.points_3d[:, 2].max() + self.points_3d[:, 2].min()) * 0.5

        ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

        # Adding arrows for the principal components
        scale = 0.1 * np.linalg.norm(self.points_3d.max(axis=0) - self.points_3d.min(axis=0))
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

    def alphashape_hull(self, alpha: float) -> Tuple[any, float]:
        """
        Compute the alpha shape of a set of 2D points. The alpha shape is a geometric structure 
        that captures the shape of a set of points in 2D space. 
        
        It is controlled by the `alpha` parameter, which determines the level of concavity of the 
        resulting shape. Smaller values of `alpha` produce more concave shapes, while larger values 
        approach the convex hull of the points.

        Parameters:
        -----------
        alpha : float
            A positive value that controls the level of concavity of the alpha shape. 
            - Lower values result in more detailed and concave shapes.
            - Higher values result in smoother and more convex shapes.
            - If `alpha` is too small, the resulting shape may become disconnected or disappear entirely.
        
        Returns:
        --------
        Tuple[Union[Polygon, MultiPolygon, None], float]
            - The computed alpha shape, which can be:
                - A `Polygon` if the shape is a single connected region.
                - A `MultiPolygon` if the shape consists of multiple disconnected regions.
                - `None` if the computation fails or the shape cannot be determined.
            - The time taken to compute the alpha shape, measured in seconds.
        """

        import time
        try:
            start_time = time.perf_counter()
            self.contour = alphashape.alphashape(self.points_2d, alpha)
            end_time = time.perf_counter()
            self.durations = end_time - start_time
            if self.contour is None:
                logger.warning("Alpha-shape computation returned None.")
                raise ValueError("Alpha-shape computation failed, try adjusting alpha.")

            logger.info("")
            logger.info("===== Computing convex hull... =====")
            return self.contour, self.durations
        except Exception as e:
            logger.error(f"An error occurred while computing the convex hull: {e}")
            raise

    def concave_hull(self, length_threshold: float, c: float) -> Tuple[np.ndarray, float]:
        """
        Compute the concave hull for a set of 2D points using a K-Nearest Neighbors (KNN) approach. 
        (See https://github.com/cubao/concave_hull) 

        The concave hull is a polygon that more accurately follows the natural boundary of a point cloud
        than the convex hull. Unlike the convex hull, which is the smallest convex polygon that encloses all
        the points, the concave hull allows for indentations and concavities, providing a closer approximation
        to the true shape of the data.

        This implementation follows the principles described in the Concaveman algorithm
        (see: https://github.com/mapbox/concaveman), and uses two main parameters to control the level
        of detail of the resulting hull
        
        Parameters:
        -----------
            - concavity: float, optional (default=1.0)
                The concavity coefficient controlling the level of detail of the hull:
                    - Values <= 1 yield a more detailed, concave shape. 
                    - Values > 1 yield a smoother, more convex shape.
            - length threshold (length_threshold): float, optional (default=0.0)
                The minimum edge length below which segments are ignored during the hull construction, which helps
                filter out edges caused by noise. The unit depends on the unit of the point cloud coordinates.

        Returns:
        --------
            - A tuple containing:
                - hull: A NumPy array of shape (m, 2) of the ordered vertices of the concave hull polygon.
                - time: A float representing the computation time in seconds.
        """

        import time
        start_time = time.perf_counter()
        hull = concave_hull(self.points_2d, concavity=c, length_threshold=length_threshold)
        hull = np.array(hull)

        if not np.allclose(hull[0], hull[-1]):  # Ensure the hull is closed
            hull = np.vstack((hull, hull[0]))

        end_time = time.perf_counter()
        self.durations = end_time - start_time
        self.contour = hull
        logger.info("")
        logger.info("===== Computing concave hull... =====")

        return self.contour, self.durations

    def compute_area(self):
        """
        Compute the area enclosed by the contour using the Shoelace formula.
        The Shoelace formula is a mathematical algorithm used to determine the area of a simple polygon
        whose vertices are described by their Cartesian coordinates in the plane.

        The equation is provided as follows:
        Area = 0.5 * |(x0 * y1 + x1 * y2 + ... + xn-1 * yn + xn * y0) - (y0 * x1 + y1 * x2 + ... + yn-1 * xn + yn * x0)|
        """

        if self.contour is None:
            raise ValueError("No contour computed, please compute a contour first.")
        try:
            logger.info("===== Computing area... =====")

            if isinstance(self.contour, Polygon):
                self.area = self.contour.area
            elif isinstance(self.contour, MultiPolygon):
                self.area = sum(p.area for p in self.contour.geoms)
            else:
                x, y = self.contour[:, 0], self.contour[:, 1]
                self.area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + (x[-1] * y[0] - x[0] * y[-1]))
            logger.info(f"Area of the contour: {self.area:.4f} m²")
        except Exception as e:
            logger.error(f"An error occurred while computing the area: {e}")
            raise

    def compute_perimeter(self):
        """
        Compute the perimeter of the contour by summing the Euclidean distances between consecutive points.
        """

        if self.contour is None:
            raise ValueError("No contour computed, please compute a contour using the computing methods implemented.")
        try:
            logger.info("")
            logger.info("===== Computing perimeter... =====")
            if isinstance(self.contour, Polygon):
                self.perimeter = self.contour.length
            elif isinstance(self.contour, MultiPolygon):
                self.perimeter = sum(p.length for p in self.contour.geoms)
            else:
                x, y = self.contour[:, 0], self.contour[:, 1]
                self.perimeter = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
            logger.info(f"Perimeter of the contour: {self.perimeter:.4f} m")
        except Exception as e:
            logger.error(f"An error occurred while computing the area: {e}")
            raise

    def compute_roughness(self) -> float:
        """
        Compute the roughness of the contour as the standard deviation of the curvature at each point of the contour.
        
        The roughness of a contour can be defined as the dispersion of the values of curvature along the contour.
        
        The curvature is computed using the formula:
        kappa = (dx * d2x - dy * d2y) / (dx^2 + dy^2)^(3/2)
        
        where: 
            - dx and dy are the first derivatives of the x and y coordinates of the contour.
            - d2x and d2y are the second derivatives of the x and y coordinates of the contour.
        
        The roughness is then calculated as the standard deviation of the curvature values. The formula is as follows:
        sigma = sqrt(1/n * sum((kappa_i - kappa_mean)^2))
        
        where:
            - n is the number of points in the contour.
            - kappa_i is the curvature at point i.
            - kappa_mean is the mean curvature of the contour.

        Returns:
        --------
            - roughness: float
                The standard deviation of the curvature values, representing the roughness of the contour.
                A high value indicates a rough contour, while a low value indicates a smooth contour.
        """

        if isinstance(self.contour, Polygon):
            x, y = self.contour.exterior.xy
        elif isinstance(self.contour, MultiPolygon):
            x, y = [], []
            for polygon in self.contour.geoms:
                xi, yi = polygon.exterior.xy
                x.extend(xi)
                y.extend(yi)
        elif isinstance(self.contour, np.ndarray):
            # Extract x and y coordinates of the contour
            x, y = self.contour[:, 0], self.contour[:, 1]
        else:
            raise TypeError("Unsupported contour format for curvature computation.")

        # Compute first derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Compute second derivatives
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # Compute curvature
        numerator = (dx * d2x - dy * d2y)
        denominator = (dx ** 2 + dy ** 2) ** (3 / 2)
        if np.any(denominator == 0):
            logger.warning(
                "!!!WARNING!!!  Denominator contains zero values, curvature may be undefined at some points.")
            denominator[denominator == 0] = np.nan
        self.curvature = numerator / denominator

        # Handle NaN values in curvature
        self.curvature[np.isnan(self.curvature)] = 0.0
        self.curvature[np.isinf(self.curvature)] = 0.0

        # Compute standard deviation of the curvature
        self.roughness = np.std(self.curvature)

        return self.roughness

    def display_contour(self):
        """
        Display the contour in the PCA plane along with the point cloud and the computed area and perimeter.
        """

        fig = plt.figure(figsize=(8, 8) if self.points_3d is None else (16, 8))

        # Adding a 3D plot if asked
        if self.points_3d is not None:
            ax3d = fig.add_subplot(121, projection='3d')
            ax3d.scatter(self.points_3d[:, 0], self.points_3d[:, 1], self.points_3d[:, 2], c='black', s=1, alpha=0.6,
                         label="Point cloud")
            ax3d.set_title("3D Point Cloud & Contour")
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.axis("equal")
            ax3d.legend()
            ax3d.set_box_aspect([1, 1, 1])
            ax2d = fig.add_subplot(122)
        else:
            ax2d = fig.add_subplot(111)

        # Calculate the area and perimeter enclosed in the contour
        self.compute_area()
        self.compute_perimeter()
        self.compute_roughness()

        if isinstance(self.contour, Polygon):
            coords = np.array(self.contour.exterior.coords)
        elif isinstance(self.contour, MultiPolygon):
            coords = []
            for poly in self.contour.geoms:
                coords.extend(poly.exterior.coords)
                coords.append((None, None))
            coords = np.array(coords, dtype=object)
        elif isinstance(self.contour, np.ndarray):
            coords = self.contour
        else:
            raise TypeError("Unsupported contour format for display.")

        # Fill the contour in the PCA plan
        polygon = plt.Polygon(coords, closed=True, facecolor='red', alpha=0.2, edgecolor='r', linewidth=2.0)
        ax2d.add_patch(polygon)
        ax2d.plot(coords[:, 0], coords[:, 1], 'r--', linewidth=2.0, label="Contour (In the PCA Plane)")
        ax2d.scatter(self.projected_points[:, 0], self.projected_points[:, 1], c='black', s=1, label="Projected points")

        # Add area and perimeter to the legend
        area_label = f"Area = {self.area:.4f} m²"
        perimeter_label = f"Perimeter = {self.perimeter:.4f} m"
        roughness_label = f"Roughness = {self.roughness:.2f}"
        ax2d.plot([], [], ' ', label=area_label)
        ax2d.plot([], [], ' ', label=perimeter_label)
        ax2d.plot([], [], ' ', label=roughness_label)

        ax2d.set_title("Contour in PCA Plane")
        ax2d.set_xlabel("PC1")
        ax2d.set_ylabel("PC2")
        ax2d.legend(loc='upper right')  # Move the legend to the top right corner
        ax2d.axis("equal")
        plt.tight_layout()
        plt.show()

    def compute_hull(self, method: str, alpha: Optional[float] = None, concavity: Optional[float] = None,
                length_threshold: Optional[float] = None):
        """
        Compute the contour of the point cloud using either the convex or concave hull method.

        Parameters:
        -----------
            - method : str, optional (default='concave')
                The method used to extract the contour. Choose between 'alphashape' or 'concave'.

            - alpha : float, optional (default=3.5), MANDATORY for alphashape method
                The alpha parameter controlling the level of concavity in the alphashape_hull method.

            - concavity : float, optional (default=None), MANDATORY for concave method
                The concavity coefficient controlling the level of detail of the hull in the concave hull method.
                
            - length_threshold : float, optional (default=None), MANDATORY for concave method
                The minimum edge length below which segments are ignored during the hull construction,
                which helps filter out edges caused by noise.
        """

        if method == 'alphashape':
            if alpha is None:
                raise ValueError("Alpha parameter must be provided for convex method.")
            self.alphashape_hull(alpha=alpha)
        elif method == 'concave':
            if concavity is None or length_threshold is None:
                raise ValueError("Concavity and length_threshold must be provided for concave method.")
            self.concave_hull(c=concavity, length_threshold=length_threshold)
        else:
            raise ValueError("Invalid method. Please choose either 'convex' or 'concave'.")

        logger.info("Hull extraction completed.")
        logger.info(f"Hull computing time: {self.durations:.4f} seconds")
        logger.info("")

    def save_contour(self):
        """
        Save the computed contour using a save file dialog.
        Formats supported: csv, vtk
        """
        if self.contour is None:
            raise ValueError("No contour has been computed yet.")

        filetypes = [
            ("CSV", "*.csv"),
            ("VTK", "*.vtk"),
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Contour As...",
            defaultextension=".vtk",
            filetypes=filetypes
        )

        if not filepath:
            print("Saving cancelled.")
            return

        ext = os.path.splitext(filepath)[1].lower()

        # Convert shapely Polygon/MultiPolygon to numpy coords
        if isinstance(self.contour, (Polygon, MultiPolygon)):
            coords = np.array(self.contour.exterior.coords)
        elif isinstance(self.contour, np.ndarray):
            coords = self.contour
        else:
            raise TypeError("Contour format not supported.")

        if ext == ".csv":
            np.savetxt(filepath, coords, delimiter=",", header="X,Y", comments='')
            print(f"Saved as CSV: {filepath}")

        elif ext == ".vtk":
            points = vtk.vtkPoints()
            polyline = vtk.vtkPolyLine()
            polyline.GetPointIds().SetNumberOfIds(len(coords))
            for i, (x, y) in enumerate(coords):
                points.InsertNextPoint(x, y, 0.0)
                polyline.GetPointIds().SetId(i, i)
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(polyline)
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetLines(cells)
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(filepath)
            writer.SetInputData(poly_data)
            writer.Write()
            print(f"Saved as VTK: {filepath}")

        else:
            print("Unsupported format.")

    def update_contour(self):
        """
        Open a persistent window for updating contour parameters, allowing iterative recomputation and final save.
        """

        def on_submit():
            try:
                method = method_var.get()
                alpha = float(alpha_entry.get()) if method == "alphashape" else None
                concavity = float(concavity_entry.get()) if method == "concave" else None
                length_threshold = float(length_threshold_entry.get()) if method == "concave" else None

                voxel_size_input = voxel_size_entry.get()
                if not voxel_size_input.replace('.', '', 1).isdigit():
                    raise ValueError("Voxel size must be a numeric value.")
                voxel_size = float(voxel_size_input)
                if voxel_size <= 0:
                    raise ValueError("Voxel size must be a positive number.")

                plt.close('all')
                self.downsample(voxel_size=voxel_size)
                self.pca_projection()
                self.compute_hull(method=method, alpha=alpha, concavity=concavity, length_threshold=length_threshold)
                self.display_contour()

            except Exception as e:
                messagebox.showerror("Error", f"Error during contour update: {e}")

        def on_save():
            if self.contour is not None:
                self.save_contour()
            else:
                messagebox.showwarning("Warning", "No valid contour computed yet.")

        def toggle_fields(*args):
            if method_var.get() == "alphashape":
                alpha_entry.config(state="normal")
                concavity_entry.config(state="disabled")
                length_threshold_entry.config(state="disabled")
            elif method_var.get() == "concave":
                alpha_entry.config(state="disabled")
                concavity_entry.config(state="normal")
                length_threshold_entry.config(state="normal")


        # Main window
        param_window = tk.Toplevel()
        param_window.title("Update Contour Parameters")
        param_window.attributes("-topmost", True)
        param_window.configure(padx=20, pady=20)
        param_window.columnconfigure(0, weight=1)

        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TFrame", background="white")

        # Frame
        frame = ttk.LabelFrame(param_window, text="Contour Parameters", padding=(15, 10))
        frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Widgets
        ttk.Label(frame, text="Method:").grid(row=0, column=0, sticky="w", pady=5)
        method_var = tk.StringVar(value="concave")
        method_combo = ttk.Combobox(frame, textvariable=method_var, values=["alphashape", "concave"], state="readonly")
        method_combo.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Alpha:").grid(row=1, column=0, sticky="w", pady=5)
        alpha_entry = ttk.Entry(frame)
        alpha_entry.insert(0, "0.05")
        alpha_entry.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Concavity:").grid(row=2, column=0, sticky="w", pady=5)
        concavity_entry = ttk.Entry(frame)
        concavity_entry.insert(0, "1.0")
        concavity_entry.grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Length Threshold:").grid(row=3, column=0, sticky="w", pady=5)
        length_threshold_entry = ttk.Entry(frame)
        length_threshold_entry.insert(0, "0.02")
        length_threshold_entry.grid(row=3, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Voxel Size:").grid(row=4, column=0, sticky="w", pady=5)
        voxel_size_entry = ttk.Entry(frame)
        voxel_size_entry.insert(0, "0.1")
        voxel_size_entry.grid(row=4, column=1, sticky="ew", pady=5)

        # Make entries expand if window is resized
        for i in range(2):
            frame.columnconfigure(i, weight=1)

        method_var.trace_add("write", toggle_fields)
        toggle_fields()

        # Buttons
        button_frame = ttk.Frame(param_window)
        button_frame.grid(row=1, column=0, pady=15, sticky="ew")

        update_button = ttk.Button(button_frame, text="Update Contour", command=on_submit)
        update_button.grid(row=0, column=0, padx=5)

        save_button = ttk.Button(button_frame, text="Save", command=on_save)
        save_button.grid(row=0, column=1, padx=5)

        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        param_window.transient()
        param_window.grab_set()
        param_window.lift()
        param_window.wait_window()
    
def extract_contour(method: str, voxel_size: float, alpha: float, concavity: float, length_threshold: float, 
                    diagnose: bool = False, visualize: bool = False):

    root = Tk()
    root.withdraw()
    cloud = ContourExtractor()
    cloud.load_cloud()
    cloud.downsample(voxel_size=voxel_size)
    cloud.pca_projection(diagnosis=diagnose, visualize=visualize)
    cloud.compute_hull(method=method, alpha=alpha, concavity=concavity, length_threshold=length_threshold)
    cloud.display_contour()
    cloud.update_contour()
    

if __name__ == "__main__":
    
    voxel_size = 0.1
    method = 'concave'  # 'alphashape' or 'concave'

    alpha = 0.05
    concavity = 1.0
    length_threshold = 0.02

    diagnose = False
    visualize = False

    extract_contour(method=method, voxel_size=voxel_size, alpha=alpha, concavity=concavity, length_threshold=length_threshold)    