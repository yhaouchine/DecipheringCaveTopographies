import numpy as np
import alphashape
from typing import Tuple, Optional
from open3d.cpu.pybind.geometry import PointCloud
from concave_hull import concave_hull
from process_cloud import display, import_cloud, pca_projection


class ContourExtractor:
    def __init__(self, voxel_size: float = 0.5, pc_name: str = None, parent_folder: str = None):
        self.voxel_size = voxel_size
        self.pc_name = pc_name
        self.parent_folder = parent_folder
        self.original_cloud: Optional[PointCloud] = None
        self.reduced_cloud: Optional[PointCloud] = None
        self.points_3d = None
        self.points_2d = None
        self.contour = None
        self.durations: Optional[float] = None

        self.load_cloud()
        self.downsample()
        self.pca_projection()

    def load_cloud(self):
        self.original_cloud, _ = import_cloud(pc_name=self.pc_name, parent_folder=self.parent_folder)

    def downsample(self):
        self.reduced_cloud = self.original_cloud.voxel_down_sample(voxel_size=self.voxel_size)
        self.points_3d = np.asarray(self.reduced_cloud.points)
        if self.points_3d.shape[0] < 3:
            raise ValueError("Not enough points to generate a contour.")

    def pca_projection(self):
        diagnosis = True
        visualize = True
        self.points_2d, _, _ = pca_projection(points_3d=self.points_3d, diagnosis=diagnosis, visualize=visualize)

    def convex_hull(self, alpha: float) -> Tuple[any, float]:
        """
        Compute the alpha shape (concave hull) of a set of 2D points.

        Parameters:
        ----------
        alpha : float
            Controls the level of concavity (lower values = more concave, higher values = more convex).
            If `alpha` is too small, the shape may be disconnected or disappear.
        
        Returns:
        -------
        A tuple containing:
            - The computed alpha shape as a `Polygon`, `MultiPolygon`, or `None` (if the computation fails).
            - The time taken to compute the shape (in seconds).        
        """
        import time
        start_time = time.perf_counter()
        self.contour = alphashape.alphashape(self.points_2d, alpha)
        end_time = time.perf_counter()
        self.durations = end_time - start_time
        return self.contour, self.durations

    def concave_hull(self, c: float = 1.0, length_threshold: float = 0.0) -> Tuple[np.ndarray, float]:
        """
            Compute the concave hull for a set of 2D points using a K-Nearest Neighbors (KNN)
            approach based on the Concaveman algorithm.

            The concave hull is a polygon that more accurately follows the natural boundary of a point cloud
            than the convex hull. Unlike the convex hull, which is the smallest convex polygon that encloses all
            the points, the concave hull allows for indentations and concavities, providing a closer approximation
            to the true shape of the data.

            This implementation follows the principles described in the Concaveman algorithm
            (see: https://github.com/mapbox/concaveman), and uses two main parameters to control the level
            of detail of the resulting hull
            
            Parameters:
            ----------
            concavity : float, optional (default=1.0)
                The concavity coefficient controlling the level of detail of the hull:
                - Lower values yield a more detailed, concave shape.
                - Higher values yield a smoother, more convex shape.
            length threshold (length_threshold) : float, optional (default=0.0)
                The minimum edge length below which segments are ignored during the hull construction,
                which helps filter out edges caused by noise.

            Returns:
            -------
            A tuple containing:
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

        return self.contour, self.durations

    def display_contour(self):
        if self.contour is None:
            raise ValueError("No contour computed, please compute a contour using the computing methods implemented.")

        if isinstance(self.contour, np.ndarray):
            contour_2d = self.contour
        else:
            x, y = self.contour.exterior.xy
            contour_2d = np.column_stack((x, y))
        display(pts=self.points_3d, contour2d=contour_2d, projected_pts=self.points_2d)

    def extract(self, method: str = 'concave'):
        if method == 'convex':
            self.convex_hull(alpha=3.5)
        elif method == 'concave':
            self.concave_hull(c=1.0, length_threshold=0.0)
        else:
            raise ValueError(f"Method {method} not supported")

        print(f"Contour computing time: {self.durations:.2f} secondes")
        self.display_contour()


if __name__ == "__main__":
    cloud = ContourExtractor(pc_name="cross_section_3_45d_clean.ply", parent_folder="saved_clouds")
    
    cloud.extract(method='concave')
