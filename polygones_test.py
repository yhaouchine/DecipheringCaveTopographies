from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from contour_extractor import ContourExtractor
from process_cloud import PointCloudProcessor
from typing import Tuple


class Shape:
    def __init__(self, nb_points, default_color=(0, 0, 0)):
        self.nb_points = nb_points
        self.color = default_color
        self.pc = None
        self.points = []
        self.x = []
        self.y = []


    def generate(self):
        pass

    def to_point_cloud(self):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        pc.colors = o3d.utility.Vector3dVector(np.tile(self.color, (len(self.points), 1)))
        return pc
    
    def save_ply(self, filename):
        o3d.io.write_point_cloud(filename, self.pc)

class Sphere(Shape):
    def __init__(self, radius, nb_points, color=(0, 0, 1)):
        super().__init__(nb_points, color)
        self.nb_points = nb_points
        self.center = (0, 0, 0)
        self.radius = radius
        self.disk_area = np.pi * (self.radius ** 2)
        self.disk_perimeter = 2 * np.pi * self.radius
        self.disk_pc = None

        self.generate()
        self.pc = self.to_point_cloud()
        print("===== Disk =====")
        print(f"Disk area = {self.disk_area:.4f}")
        print(f"Disk perimeter = {self.disk_perimeter:.4f}")
        print()

    def generate(self):
        phi = np.random.uniform(0, np.pi, self.nb_points)
        theta = np.random.uniform(0, 2 * np.pi, self.nb_points)
        r = np.cbrt(np.random.uniform(0, self.radius, self.nb_points))  # Cube root for uniform distribution in volume

        x = self.center[0] + r * np.sin(phi) * np.cos(theta)
        y = self.center[1] + r * np.sin(phi) * np.sin(theta)
        z = self.center[2] + r * np.cos(phi)

        self.points = np.column_stack((x, y, z))
    
    def extract_disk(self, tolerance=0.01):
        # Extract points close to the center of the sphere (z = 0)
        z_level = 0
        disk_points = self.points[np.abs(self.points[:, 2] - z_level) < tolerance]
        
        # Create a new point cloud for the disk
        self.disk_pc = o3d.geometry.PointCloud()
        self.disk_pc.points = o3d.utility.Vector3dVector(disk_points)
        self.disk_pc.colors = o3d.utility.Vector3dVector(np.tile(self.color, (len(disk_points), 1)))
        
        return self.disk_pc

class Cuboid(Shape):
    def __init__(self, nb_points, lengths=(4, 2, 3), center=(0, 0, 0), color=(1, 0, 0)):
        super().__init__(nb_points, color)
        self.lengths = np.array(lengths)
        self.center = np.array(center)
        self.nb_points = nb_points
        self.rectangle_area = self.lengths[0] * self.lengths[1]
        self.rectangle_perimeter = 2 * (self.lengths[0] + self.lengths[1])
        self.rectangle_pc = None

        self.generate()
        self.pc = self.to_point_cloud()
        print("===== Rectangle =====")
        print(f"Rectangle area = {self.rectangle_area:.4f}")
        print(f"Rectangle perimeter =  {self.rectangle_perimeter:.4f}")
        print()

    def generate(self):
        half_lengths = self.lengths / 2
        x_min, y_min, z_min = self.center - half_lengths
        x_max, y_max, z_max = self.center + half_lengths

        x = np.random.uniform(x_min, x_max, self.nb_points)
        y = np.random.uniform(y_min, y_max, self.nb_points)
        z = np.random.uniform(z_min, z_max, self.nb_points)

        self.points = np.column_stack((x, y, z))

    def extract_section(self):
        z_level = 0
        tolerance = 0.01
        rectangle_points = self.points[np.abs(self.points[:, 2] - z_level) < tolerance]

        # Create a new point cloud for the section
        self.rectangle_pc = o3d.geometry.PointCloud()
        self.rectangle_pc.points = o3d.utility.Vector3dVector(rectangle_points)
        self.rectangle_pc.colors = o3d.utility.Vector3dVector(np.tile(self.color, (len(rectangle_points), 1)))

        return self.rectangle_pc

class Pyramid(Shape):
    def __init__(self, nb_points, base_size=4, height=5, center=(0, 0, 0), color=(0, 1, 0)):
        super().__init__(nb_points, color)
        self.base_size = base_size
        self.height = height
        self.nb_points = nb_points
        self.real_area = 0.5 * self.base_size * self.height
        self.real_perimeter = self.base_size + 2 * np.sqrt((self.base_size / 2) ** 2 + self.height ** 2)
        self.center = np.array(center)
        self.triangle_pc = None

        self.generate()
        self.pc = self.to_point_cloud()
        print("===== Triangle =====")
        print(f"Triangle area = {self.real_area: .4f}")
        print(f"Triangle perimeter = {self.real_perimeter: .4f}")
        print()

    def generate(self):
        cx, cy, cz = self.center
        half_size = self.base_size / 2

        # Pyramid vertices
        v_base = [
            (cx - half_size, cy - half_size, cz),
            (cx + half_size, cy - half_size, cz),
            (cx + half_size, cy + half_size, cz),
            (cx - half_size, cy + half_size, cz)
        ]
        v_top = (cx, cy, cz + self.height)

        # Generate random points inside the pyramid
        points = []
        for _ in range(self.nb_points):
            # Generate a random height level with linear distribution
            h = np.random.uniform(0, 1)
            z = cz + h * self.height

            # Compute the size of the base at this height
            scale = 1 - h
            base_half_size = half_size * scale

            # Generate a random point within the scaled base
            x = np.random.uniform(cx - base_half_size, cx + base_half_size)
            y = np.random.uniform(cy - base_half_size, cy + base_half_size)

            points.append((x, y, z))

        self.points = np.array(points)

    def extract_section(self):
        x_level = 5.0
        tolerance = 0.01
        triangle_points = self.points[np.abs(self.points[:, 0] - x_level) < tolerance]

        self.triangle_pc = o3d.geometry.PointCloud()
        self.triangle_pc.points = o3d.utility.Vector3dVector(triangle_points)
        self.triangle_pc.colors = o3d.utility.Vector3dVector(np.tile(self.color, (len(triangle_points), 1)))

        return self.triangle_pc
    
def generate_save_clouds(
    num_points: int, 
    sphere_radius: float, 
    save_folder: str, 
    save: bool = False, 
    s: bool = True,
    c: bool = True,
    p: bool = True
    ) -> Tuple[Sphere, Cuboid, Pyramid]:
    
    # Generating shapes
    sphere_instance = Sphere(radius=sphere_radius, nb_points=num_points, color=(0, 0, 1)) if s else None
    cuboid_instance = Cuboid(nb_points=num_points, lengths=(4, 2, 3), color=(1, 0, 0)) if c else None
    pyramid_instance = Pyramid(nb_points=num_points, base_size=4, height=5, center=(5, 5, 0), color=(0, 1, 0)) if p else None

    if save:
        try:
            save_folder = Path(save_folder) if not isinstance(save_folder, Path) else save_folder
            if s:
                sphere_instance.save_ply(save_folder / "sphere.ply")
            if c:
                cuboid_instance.save_ply(save_folder / "cuboid.ply")
            if p:
                pyramid_instance.save_ply(save_folder / "pyramid.ply")

        except Exception as e:
            import traceback
            print(f"An error occurred while saving PLY files: {e}")
            traceback.print_exc()

    return sphere_instance, cuboid_instance, pyramid_instance

if __name__ == "__main__":
        
        save_folder = Path("test_shapes")
        save_folder.mkdir(parents=True, exist_ok=True)

        sphere_instance, cuboid_instance, pyramid_instance = generate_save_clouds(
            num_points=20000, 
            sphere_radius=1, 
            save=False, 
            save_folder=save_folder, 
            s=True, 
            c=True, 
            p=True
            )
        
        # ======= Generate disk =======
        # disk_pc = sphere_instance.extract_disk()
        # disk_processor = PointCloudProcessor(sphere_instance.pc)
        # disk_processor.visualizer(window_name="Disk", geom1=disk_pc, save=False)

        # ======= Generate rectangle =======
        # parallelogram_pc = cuboid_instance.extract_section()
        # parallelogram_processor = PointCloudProcessor(cuboid_instance.pc)
        # parallelogram_processor.visualizer(window_name="Parallelogram", geom1=parallelogram_pc, save=False)

        # ======= Generate triangle =======
        # triangle_pc = pyramid_instance.extract_section()
        # triangle_processor = PointCloudProcessor(pyramid_instance.pc)
        # triangle_processor.visualizer(window_name="Triangle", geom1=triangle_pc, save=False)

        # cloud = ContourExtractor()
        # cloud.load_cloud(pc_name="triangle.ply", parent_folder=save_folder)
        # cloud.downsample(voxel_size=0.01)
        # cloud.pca_projection()
        # cloud.extract(concavity=1.0, length_threshold=0.2)

    
