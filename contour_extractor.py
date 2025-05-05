import numpy as np
import alphashape
import logging
import matplotlib.pyplot as plt
import open3d as o3d
import tkinter as tk
import time
import os
import vtk
from process_cloud import PCASection, DevelopedSection
from tkinter import filedialog, Tk, messagebox, ttk
from typing import Tuple, Optional
from open3d.cpu.pybind.geometry import PointCloud
from concave_hull import concave_hull
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class UserInterface:
    def __init__(self, master, controller):
        self.master = master
        self.controller = controller

        # Tk variables
        self.file_path_var = tk.StringVar()
        self.hull_method_var = tk.StringVar(value="concave")
        self.section_type_var = tk.StringVar(value="developed")
        self.poisson_enabled_var = tk.BooleanVar(value=False)
        self.densification_enabled_var = tk.BooleanVar(value=False)
        self.densification_method_var = tk.StringVar(value="linear")

        # Build UI
        self._init_window()
        self._create_section_frame()
        self._create_contour_frame()
        self._create_poisson_frame()
        self._create_densification_frame()
        self._create_button_frame()
        self._bind_traces()
        self._set_initial_states()

        # Block until window is closed
        self.window.wait_window()

    def _init_window(self):
        self.window = tk.Toplevel(self.master)
        self.window.title("Update Parameters")
        self.window.attributes("-topmost", True)
        self.window.configure(padx=20, pady=20)
        for col in (0,1):
            self.window.columnconfigure(col, weight=1)
        self.window.transient()
        self.window.grab_set()
        self.window.lift()

    def _create_section_frame(self):
        frame = ttk.LabelFrame(self.window, text="Section Parameters", padding=(15,10))
        frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
        for col in (0,1): frame.columnconfigure(col, weight=1)

        ttk.Label(frame, text="Point cloud file:").grid(row=0, column=0, sticky="w", pady=5)
        self.file_path_entry = ttk.Entry(frame, textvariable=self.file_path_var, state="readonly")
        Tooltip(self.file_path_entry, "Select a point cloud file to load.")

        browse_button = ttk.Button(frame, text="Browse", command=self.controller.load_cloud)
        browse_button.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(frame, text="Section type:").grid(row=1, column=0, sticky="w", pady=5)
        self.section_type_combo = ttk.Combobox(frame, textvariable=self.section_type_var,
                                         values=["developed", "pca"], state="readonly")
        self.section_type_combo.grid(row=1, column=1, sticky="ew", pady=5)
        Tooltip(
            self.section_type_combo,
            "Select the type of section to compute. \n" \
            "- Developed: Develop the section along its previously interpolated line. REQUIREMENT: *_interpolated_line.npy file. \n" \
            "- PCA: Project the points in the principal component plan after performing and principal component analysis (PCA)."
            )

    def _create_contour_frame(self):
        frame = ttk.LabelFrame(self.window, text="Contour Parameters", padding=(15,10))
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        for col in (0,1): frame.columnconfigure(col, weight=1)

        ttk.Label(frame, text="Contour Method:").grid(row=0, column=0, sticky="w", pady=5)
        self.hull_method_combo = ttk.Combobox(frame, textvariable=self.hull_method_var,
                                         values=["alphashape", "concave"], state="readonly")
        self.hull_method_combo.grid(row=0, column=1, sticky="ew", pady=5)
        Tooltip(
            self.hull_method_combo,
            "Select the method to compute the contour. \n" \
            "- Alphashape: Compute the contour using the alpha shape method. \n" \
            "- Concave: Compute the contour using the concave hull method."
        )

        ttk.Label(frame, text="Alpha:").grid(row=1, column=0, sticky="w", pady=5)
        self.alpha_entry = ttk.Entry(frame); self.alpha_entry.insert(0, "0.05")
        self.alpha_entry.grid(row=1, column=1, sticky="ew", pady=5)
        Tooltip(
            self.alpha_entry,
            "Controls the level of concavity of the contour. \n" \
            "Alpha ranges from 0 (convex shape) to +infinity (concave shape), Although high values (>10) might cause irrelevent/unwanted behaviour. " \
        )


        ttk.Label(frame, text="Concavity:").grid(row=2, column=0, sticky="w", pady=5)
        self.concavity_entry = ttk.Entry(frame); self.concavity_entry.insert(0, "1.0")
        self.concavity_entry.grid(row=2, column=1, sticky="ew", pady=5)
        Tooltip(
            self.concavity_entry,
            "Controls the level of concavity of the contour. \n" \
            "Value = 1 yield a detailed, concave shape. Value > 1 yield a smoother, more convex shape. Value < 1 are possible to obtain a very concave shapes" \
            " although it can lead to an overly detailed or disconnected shape."
        )

        ttk.Label(frame, text="Length Threshold:").grid(row=3, column=0, sticky="w", pady=5)
        self.length_threshold_entry = ttk.Entry(frame); self.length_threshold_entry.insert(0, "0.02")
        self.length_threshold_entry.grid(row=3, column=1, sticky="ew", pady=5)
        Tooltip(
            self.length_threshold_entry,
            "The minimum edge length below which segments are ignored during the hull construction, which helps filter out edges caused by noise." \
            "The unit depends on the unit of the point cloud coordinates (usually meters)."
        )

        ttk.Label(frame, text="Voxel Size:").grid(row=4, column=0, sticky="w", pady=5)
        self.voxel_size_entry = ttk.Entry(frame); self.voxel_size_entry.insert(0, "0.1")
        self.voxel_size_entry.grid(row=4, column=1, sticky="ew", pady=5)
        Tooltip(
            self.voxel_size_entry,
            "The size of a voxel of the grid used for downsampling the contour point cloud. The unit correspond to the unit of the point cloud coordinates (usually meters)."
        )

    def _create_poisson_frame(self):
        frame = ttk.LabelFrame(self.window, text="Poisson Parameters", padding=(15,10))
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        for col in (0,1): frame.columnconfigure(col, weight=1)

        poisson_checkbutton = ttk.Checkbutton(frame, text="Poisson reconstruction [EXPERIMENTAL]",
                                              variable=self.poisson_enabled_var,
                                              command=self._toggle_poisson_fields)
        poisson_checkbutton.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))
        Tooltip(
            poisson_checkbutton,
            "Enable Poisson reconstruction to fill holes (i.e. areas where points are missing) in the contour." 
        )

        ttk.Label(frame, text="Depth:").grid(row=1, column=0, sticky="w", pady=5)
        self.depth_entry = ttk.Entry(frame); self.depth_entry.insert(0, "8")
        self.depth_entry.grid(row=1, column=1, sticky="ew", pady=5)
        Tooltip(
            self.depth_entry,
            "The octree depth used for Poisson reconstruction."
        )

        ttk.Label(frame, text="Scale:").grid(row=2, column=0, sticky="w", pady=5)
        self.scale_entry = ttk.Entry(frame); self.scale_entry.insert(0, "2.0")
        self.scale_entry.grid(row=2, column=1, sticky="ew", pady=5)
        Tooltip(
            self.scale_entry,
            "The scale factor used for poisson reconstruction."
        )

        ttk.Label(frame, text="Density threshold:").grid(row=3, column=0, sticky="w", pady=5)
        self.density_threshold_entry = ttk.Entry(frame); self.density_threshold_entry.insert(0, "0.02")
        self.density_threshold_entry.grid(row=3, column=1, sticky="ew", pady=5)
        Tooltip(
            self.density_threshold_entry,
            "The density threshold used for poisson reconstruction."
        )

        ttk.Label(frame, text="Number of points:").grid(row=4, column=0, sticky="w", pady=5)
        self.nb_points_entry = ttk.Entry(frame); self.nb_points_entry.insert(0, "5000")
        self.nb_points_entry.grid(row=4, column=1, sticky="ew", pady=5)
        Tooltip(
            self.nb_points_entry,
            "The target number of points used for poisson reconstruction."
        )

    def _create_densification_frame(self):
        frame = ttk.LabelFrame(self.window, text="Densification Parameters", padding=(15,10))
        frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)
        for col in (0,1): frame.columnconfigure(col, weight=1)

        densification_checkbutton = ttk.Checkbutton(frame, text="Contour densification",
                                                    variable=self.densification_enabled_var,
                                                    command=self._toggle_densification_fields)
        densification_checkbutton.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0,10))
        Tooltip(
            densification_checkbutton,
            "Enable contour densification to interpolate points in areas where points are missing."
        )

        ttk.Label(frame, text="Densification Method:").grid(row=1, column=0, sticky="w", pady=5)
        self.densification_method_combo = ttk.Combobox(
            frame, textvariable=self.densification_method_var,
            values=["linear", "cubicspline"], state="disabled"
        )
        self.densification_method_combo.grid(row=1, column=1, sticky="ew", pady=5)
        self.densification_method_combo.bind(
            "<<ComboboxSelected>>", lambda e: self._toggle_densification_fields()
        )
        Tooltip(
            self.densification_method_combo,
            "Select the method to densify the contour. \n" \
            "- Linear: Interpolate points using a linear method. \n" \
            "- Cubic Spline: Interpolate points using a cubic spline method."
        )

        ttk.Label(frame, text="Max segment length:").grid(row=2, column=0, sticky="w", pady=5)
        self.segment_length_entry = ttk.Entry(frame); self.segment_length_entry.insert(0, "0.01")
        self.segment_length_entry.grid(row=2, column=1, sticky="ew", pady=5)
        Tooltip(
            self.segment_length_entry,
            "The maximum length of the segments constructed by the densification process. " \
            " The unit correspond to the unit of the point cloud coordinates (usually meters)."
        )

        ttk.Label(frame, text="Max hole gap:").grid(row=3, column=0, sticky="w", pady=5)
        self.hole_gap_entry = ttk.Entry(frame); self.hole_gap_entry.insert(0, "1.0")
        self.hole_gap_entry.grid(row=3, column=1, sticky="ew", pady=5)
        Tooltip(
            self.hole_gap_entry,
            "The maximum gap between points in the contour that will be filled during the densification process."
        )

        ttk.Label(frame, text="Window:").grid(row=4, column=0, sticky="w", pady=5)
        self.window_entry = ttk.Entry(frame); self.window_entry.insert(0, "3")
        self.window_entry.grid(row=4, column=1, sticky="ew", pady=5)
        Tooltip(
            self.window_entry,
            "The size of the window within which the points will be taken into account for the densification process. " \
            "TO BE USED WITH CUBIC SPLINE METHOD ONLY."
        )

    def _create_button_frame(self):
        frame = ttk.Frame(self.window)
        frame.grid(row=4, column=0, columnspan=2, pady=15, sticky="ew")
        for col in (0,1): frame.columnconfigure(col, weight=1)

        ttk.Button(frame, text="Compute", command=self._on_submit).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text="Save", command=self._on_save).grid(row=0, column=1, padx=5)

    def _bind_traces(self):
        # Bind trace events to toggle fields based on user input
        
        self.hull_method_var.trace_add("write", lambda *args: self._toggle_hull_method_fields())
        self.poisson_enabled_var.trace_add("write", lambda *args: self._toggle_poisson_fields())
        self.densification_enabled_var.trace_add("write", lambda *args: self._toggle_densification_fields())

    def _set_initial_states(self):
        self._toggle_hull_method_fields()
        self._toggle_poisson_fields()
        self._toggle_densification_fields()

    def _toggle_hull_method_fields(self):
        if self.hull_method_var.get() == "alphashape":
            self.alpha_entry.config(state="normal")
            self.concavity_entry.config(state="disabled")
            self.length_threshold_entry.config(state="disabled")
        else:
            self.alpha_entry.config(state="disabled")
            self.concavity_entry.config(state="normal")
            self.length_threshold_entry.config(state="normal")

    def _toggle_poisson_fields(self):
        state = "normal" if self.poisson_enabled_var.get() else "disabled"
        self.depth_entry.config(state=state)
        self.scale_entry.config(state=state)
        self.density_threshold_entry.config(state=state)
        self.nb_points_entry.config(state=state)

    def _toggle_densification_fields(self):
        if self.densification_enabled_var.get():
            self.densification_method_combo.config(state="readonly")
            self.segment_length_entry.config(state="normal")
            self.hole_gap_entry.config(state="normal")
            if self.densification_method_var.get() == "cubicspline":
                self.window_entry.config(state="normal")
            else:
                self.window_entry.config(state="disabled")
        else:
            self.densification_method_combo.config(state="disabled")
            self.segment_length_entry.config(state="disabled")
            self.hole_gap_entry.config(state="disabled")
            self.window_entry.config(state="disabled")

    def _on_submit(self):
        try:
            # Get user inputs
            section_type = self.section_type_var.get()
            hull_method = self.hull_method_var.get()
            alpha = float(self.alpha_entry.get()) if hull_method == "alphashape" else None
            concavity = float(self.concavity_entry.get()) if hull_method == "concave" else None
            length_threshold = float(self.length_threshold_entry.get()) if hull_method == "concave" else None

            voxel_str = self.voxel_size_entry.get()
            if not voxel_str.replace('.', '', 1).isdigit():
                raise ValueError("Voxel size must be numeric.")
            voxel_size = float(voxel_str)
            if voxel_size <= 0:
                raise ValueError("Voxel size must be positive.")

            depth = int(self.depth_entry.get())
            scale = float(self.scale_entry.get())
            density = float(self.density_threshold_entry.get())
            nb_pts = int(self.nb_points_entry.get())

            dens_method = self.densification_method_var.get()
            max_gap = float(self.hole_gap_entry.get())
            max_len = float(self.segment_length_entry.get())
            window = int(self.window_entry.get())

            # Execute pipeline
            plt.close('all')
            self.controller.downsample(voxel_size=voxel_size)
            if self.poisson_enabled_var.get():
                self.controller.fill_holes_poisson(
                    depth=depth, scale=scale,
                    density_threshold_quantile=density,
                    target_number_of_points=nb_pts
                )
            
            # Develop or performe a PCA according to the section type selected
            if section_type == "developed":
                self.controller.section_type = "developed"
                developed_section = DevelopedSection(pc=self.controller.reduced_cloud)
                developed_section.section = self.controller.points_3d

                if self.controller.pc_name:
                    npy_file_path = os.path.join(self.controller.parent_folder, f"{self.controller.pc_name}_interpolated_line.npy")
                    print (f"Loading interpolated line from {npy_file_path}...")
                    if os.path.exists(npy_file_path):
                        developed_section.interpolated_line = np.load(npy_file_path)
                        logger.info(f"Interpolated line loaded from {npy_file_path}.")
                    else:
                        logger.warning(f"Interpolated line file not found: {npy_file_path}.")
                else:
                    logger.error("Point cloud name (pc_name) is not set. Cannot construct file path for interpolated line.")

                self.controller.projected_points = developed_section.compute(show=False)
            elif section_type == "pca":
                self.controller.section_type = "pca"
                projected_section = PCASection(pc=self.controller.reduced_cloud)
                projected_section.section = self.controller.points_3d
                self.controller.projected_points = projected_section.compute(show=False, diagnosis=diagnose)


            sorted_pts = sort_by_chain(self.controller.projected_points)
            if self.densification_enabled_var.get():
                self.controller.projected_points = self.controller.densify_contour(
                    contour=sorted_pts,
                    densification_method=dens_method,
                    max_segment_length=max_len,
                    max_allowed_gap=max_gap,
                    window=window
                )
            else:
                self.controller.projected_points = sorted_pts

            self.controller.compute_hull(
                hull_method=hull_method,
                alpha=alpha,
                concavity=concavity,
                length_threshold=length_threshold
            )
            self.controller.display_contour()

        except Exception as e:
            messagebox.showerror("Error during contour update", str(e))

    def _on_save(self):
        if self.controller.contour is None:
            messagebox.showwarning("Warning", "No valid contour computed yet.")
        else:
            self.controller.save_contour()

class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.id_after = None

        widget.bind("<Enter>", self.schedule_tooltip)
        widget.bind("<Leave>", self.hide_tooltip)
    
    def schedule_tooltip(self, event = None):
        self.unschedule_tooltip()
        self.id_after = self.widget.after(self.delay, self.show_tooltip)

    def unschedule_tooltip(self):
        id_after = self.id_after
        self.id_after = None
        if id_after:
            self.widget.after_cancel(id_after)

    def show_tooltip(self, event = None):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.wm_attributes("-topmost", True)
        tw.attributes("-alpha", 0.90)
        label = tk.Label(tw, text=self.text, justify='left',
                 background="#fdf6e3",  # beige doux
                 foreground="#333333",  # texte gris foncé
                 relief='solid', borderwidth=1,
                 font=("Segoe UI", 10, "normal"),
                 wraplength=500,  # pour gérer les longs textes
                 padx=5, pady=3)
        label.pack(ipadx=5, ipady=2)

    def hide_tooltip(self, event=None):
        self.unschedule_tooltip()
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()

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
        self.section_type = None

    def load_cloud(self):
        """
        Load a point cloud from a file using Open3D point cloud reading function.        
        """

        try:
            pc_path = filedialog.askopenfilename(title="Select Point Cloud file",
                                                 filetypes=[("Point Cloud files", "*.ply")])
            # on stocke aussi le dossier parent pour retrouver le .npy
            self.parent_folder = os.path.dirname(pc_path)
            self.pc_name = os.path.splitext(os.path.basename(pc_path))[0]
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
            self.contour = alphashape.alphashape(self.projected_points, alpha)
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
        hull = concave_hull(self.projected_points, concavity=c, length_threshold=length_threshold)
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
        Display the contour along with the point cloud and the computed area and perimeter.
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

        # Fill the contour
        polygon = plt.Polygon(coords, closed=True, facecolor='red', alpha=0.2, edgecolor='r', linewidth=2.0)
        ax2d.add_patch(polygon)
        ax2d.plot(coords[:, 0], coords[:, 1], 'r--', linewidth=2.0, label="Contour")
        ax2d.scatter(self.projected_points[:, 0], self.projected_points[:, 1], c='black', s=1, label="Projected points")

        # Add area and perimeter to the legend
        area_label = f"Area = {self.area:.4f} m²"
        perimeter_label = f"Perimeter = {self.perimeter:.4f} m"
        roughness_label = f"Roughness = {self.roughness:.2f}"
        ax2d.plot([], [], ' ', label=area_label)
        ax2d.plot([], [], ' ', label=perimeter_label)
        ax2d.plot([], [], ' ', label=roughness_label)

        # Change the labels according to the section type
        if self.section_type == "pca":
            ax2d.set_title("Contour in PCA Plane")
            ax2d.set_xlabel("PC1")
            ax2d.set_ylabel("PC2")
        elif self.section_type == "developed":
            ax2d.set_title("Developed Section Contour")
            ax2d.set_xlabel("X (m)")
            ax2d.set_ylabel("Z (m)")
        ax2d.legend(loc='upper right')  # Move the legend to the top right corner
        ax2d.axis("equal")
        plt.tight_layout()
        plt.show()

    def compute_hull(self, hull_method: str, alpha: Optional[float] = None, concavity: Optional[float] = None,
                length_threshold: Optional[float] = None):
        """
        Compute the contour of the point cloud using either the convex or concave hull method.

        Parameters:
        -----------
            - hull_method : str, optional (default='concave')
                The method used to extract the contour. Choose between 'alphashape' or 'concave'.

            - alpha : float, optional (default=3.5), MANDATORY for alphashape method
                The alpha parameter controlling the level of concavity in the alphashape_hull method.

            - concavity : float, optional (default=None), MANDATORY for concave method
                The concavity coefficient controlling the level of detail of the hull in the concave hull method.
                
            - length_threshold : float, optional (default=None), MANDATORY for concave method
                The minimum edge length below which segments are ignored during the hull construction,
                which helps filter out edges caused by noise.
        """

        if hull_method == 'alphashape':
            if alpha is None:
                raise ValueError("Alpha parameter must be provided for convex method.")
            self.alphashape_hull(alpha=alpha)
        elif hull_method == 'concave':
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

    def fill_holes_poisson(self,depth: int = 8,width: int = 0,scale: float = 1.0,linear_fit: bool = False,density_threshold_quantile: float = 0.02,target_number_of_points: int = 5000):
        """
        
        """
        pcd = self.reduced_cloud
        # Estimate normals
        radius = self.voxel_size * 2 if self.voxel_size is not None else 0.1
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius, max_nn=30))
        pcd.orient_normals_to_align_with_direction([0, 0, 1])

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=linear_fit
        )
        densities = np.asarray(densities)

        # Remove low-density vertices
        thresh = np.quantile(densities, density_threshold_quantile)
        mesh.remove_vertices_by_mask(densities < thresh)

        # Simplify mesh (optional)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=1000)

        # Resample points
        sampled_pcd = mesh.sample_points_poisson_disk(number_of_points=target_number_of_points)

        # Update internal cloud
        self.reduced_cloud = sampled_pcd
        self.points_3d = np.asarray(sampled_pcd.points)
        
    def densify_contour(self, contour: np.ndarray, densification_method, max_segment_length: float = 0.01, max_allowed_gap: float = 1.0, window: int = 2) -> np.ndarray:
        """
        Interpolates points strictly between each pair of consecutive points in a 2D open contour.
        Only densifies segments longer than `max_segment_length` and shorter than `max_allowed_gap`.

        Parameters:
            contour (np.ndarray): (N, 2) array of ordered 2D contour points (no wrap-around).
            max_segment_length (float): Segment length above which points will be interpolated.
            max_allowed_gap (float): Max distance to allow interpolation; beyond this, segment is skipped.

        Returns:
            np.ndarray: Densified contour.
        """
        densified = [contour[0]]

        for i in range(1, len(contour)):
            p1 = contour[i - 1]
            p2 = contour[i]
            segment_length = np.linalg.norm(p2 - p1)

            # On ne densifie que si c'est assez long mais pas trop (trou)
            if max_segment_length < segment_length < max_allowed_gap:
                n_interp = int(np.ceil(segment_length / max_segment_length)) - 1
                t_values = np.linspace(0, 1, n_interp + 2)[1:-1]
                if densification_method == 'cubicspline':
                    interp_pts = spline_interpolate_segment(contour, i-1, i, n_points=n_interp + 2, window=window)
                elif densification_method == 'linear':
                    interp_pts = (1 - t_values)[:, None] * p1 + t_values[:, None] * p2
                densified.extend(interp_pts)

            densified.append(p2)

        self.projected_points = np.array(densified)
        return self.projected_points

def spline_interpolate_segment(contour: np.ndarray, i1: int, i2: int, n_points: int = 10, window: int = 3) -> np.ndarray:
    """
    Interpolation entre deux points d’un contour à l’aide d’une spline construite localement.

    i1, i2 : indices dans le contour
    window : nombre de points avant et après à prendre pour la spline
    """
    from scipy.interpolate import CubicSpline
    i_start = max(0, i1 - window)
    i_end = min(len(contour), i2 + window + 1)

    segment = contour[i_start:i_end]
    if len(segment) < 4:
        return np.linspace(contour[i1], contour[i2], n_points)[1:-1]  # fallback linéaire

    x = np.arange(len(segment))
    cs_x = CubicSpline(x, segment[:, 0])
    cs_y = CubicSpline(x, segment[:, 1])

    x_interp = np.linspace(window, len(segment) - window - 1, n_points)
    x_vals = cs_x(x_interp)
    y_vals = cs_y(x_interp)

    return np.stack((x_vals, y_vals), axis=1)[1:-1]  # on enlève p1 et p2 pour éviter duplicata

def sort_by_chain(contour: np.ndarray) -> np.ndarray:
    used = np.zeros(len(contour), bool)
    tree = KDTree(contour)
    # point de départ : celui qui a la plus petite abscisse (par exemple)
    idx0 = np.argmin(contour[:,0])
    order = [idx0]
    used[idx0] = True

    # on “marche” en choisissant à chaque fois le plus proche voisin non utilisé
    for _ in range(len(contour)-1):
        dist, idxs = tree.query(contour[order[-1]], k= len(contour))
        for d, j in zip(dist, idxs):
            if not used[j]:
                order.append(j)
                used[j] = True
                break

    return contour[order]

def extract_contour():
    root = Tk()
    root.withdraw()
    cloud = ContourExtractor()
    UserInterface(master=root, controller=cloud)


if __name__ == "__main__":
    voxel_size = 0.01
    section_type = 'developed'  # 'pca' or 'developed'
    hull_method = 'concave'  # 'alphashape' or 'concave'
    alpha = 0.05
    concavity = 1.0
    length_threshold = 0.02
    diagnose = False
    visualize = False

    extract_contour()