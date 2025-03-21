from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from contour_extractor import ContourExtractor
from process_cloud import PointCloudProcessor


class Shape:
    def __init__(self, nb_points, color = (0, 0, 0)):
        self.color = color
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
        self.real_area = np.pi * (self.radius ** 2)
        self.real_perimeter = 2 * np.pi * self.radius
        self.disk_pc = None

        self.generate()
        self.pc = self.to_point_cloud()

    def generate(self):
        phi = np.random.uniform(0, np.pi, self.nb_points)
        theta = np.random.uniform(0, 2 * np.pi, self.nb_points)
        r = np.random.uniform(0, self.radius, self.nb_points) ** (1/3)  # Cube root for uniform distribution in volume

        x = self.center[0] + r * np.sin(phi) * np.cos(theta)
        y = self.center[1] + r * np.sin(phi) * np.sin(theta)
        z = self.center[2] + r * np.cos(phi)

        self.points = np.column_stack((x, y, z))
    
    def extract_disk(self):
        # Extract points close to the center of the sphere (z = 0)
        z_level = 0
        tolerance = 0.005
        disk_points = self.points[np.abs(self.points[:, 2] - z_level) < tolerance]
        
        # Create a new point cloud for the disk
        self.disk_pc = o3d.geometry.PointCloud()
        self.disk_pc.points = o3d.utility.Vector3dVector(disk_points)
        self.disk_pc.colors = o3d.utility.Vector3dVector(np.tile(self.color, (len(disk_points), 1)))
        
        return self.disk_pc

class Cuboid(Shape):
    def __init__(self, nb_points, min_corner=(-2, -1, -1), max_corner=(2, 1, 1), color=(1, 0, 0)):
        super().__init__(nb_points, color)
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.nb_points = nb_points
        self.real_area = None
        self.real_perimeter = None
        self.generate()
        self.pc = self.to_point_cloud()
    
    def generate(self):
        x_min, y_min, z_min = self.min_corner
        x_max, y_max, z_max = self.max_corner

        x = np.random.uniform(x_min, x_max, self.nb_points)
        y = np.random.uniform(y_min, y_max, self.nb_points)
        z = np.random.uniform(z_min, z_max, self.nb_points)

        choice = np.random.choice([0, 1, 2], size=self.nb_points)
        x[choice == 0] = np.random.choice([x_min, x_max], size=np.sum(choice == 0))
        y[choice == 1] = np.random.choice([y_min, y_max], size=np.sum(choice == 1))
        z[choice == 2] = np.random.choice([z_min, z_max], size=np.sum(choice == 2))

        self.points = np.column_stack((x, y, z))

class Pyramid(Shape):
    def __init__(self, nb_points, base_size=4, height=5, center=(0, 0, 0), color=(0, 1, 0)):
        super().__init__(nb_points, color)
        self.base_size = base_size
        self.height = height
        self.nb_points = nb_points
        self.real_area = None
        self.real_perimeter = None
        self.center = np.array(center)
        self.generate()
        self.pc = self.to_point_cloud()

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

        base_points_count = self.nb_points // 2
        side_points_count = self.nb_points // 2

        # Generate the base points
        x = np.random.uniform(cx - half_size, cx + half_size, base_points_count)
        y = np.random.uniform(cy - half_size, cy + half_size, base_points_count)
        z = np.full_like(x, cz)
        base_points = np.column_stack((x, y, z))

        # Generate the faces points
        side_points = []
        for i in range(4):
            p1, p2 = v_base[i], v_base[(i + 1) % 4]
            for _ in range(side_points_count // 4):
                t1, t2 = np.random.uniform(0, 1, 2)
                if t1 + t2 > 1:
                    t1, t2 = 1 - t1, 1 - t2
                point = (1 - t1 - t2) * np.array(p1) + t1 * np.array(p2) + t2 * np.array(v_top)
                side_points.append(point)

        side_points = np.array(side_points)
        self.points = np.vstack((base_points, side_points))


if __name__ == "__main__":
        n_points = 2000000  # Reduced number of points for testing
        save_folder = Path("test_shapes")
        save_folder.mkdir(parents=True, exist_ok=True)

        sphere_instance = Sphere(radius=1, nb_points=n_points, color=(0, 0, 1))
        cuboid_instance = Cuboid(nb_points=n_points, min_corner=(-2, -1, -1), max_corner=(2, 1, 1), color=(1, 0, 0))
        pyramid_instance = Pyramid(nb_points=n_points, base_size=4, height=5, center=(5, 5, 0), color=(0, 1, 0))

        # # Saving files
        # try:
        #     sphere_instance.save_ply(save_folder / "sphere.ply")
        #     cuboid_instance.save_ply(save_folder / "cuboid.ply")
        #     pyramid_instance.save_ply(save_folder / "pyramid.ply")
        #     print("PLY files saved.")
        # except Exception as e:
        #     print(f"An error occurred while saving PLY files: {e}")
        
        # disk_pc = sphere_instance.extract_disk()
        # disk_processor = PointCloudProcessor(sphere_instance.pc)
        # disk_processor.visualizer(window_name="Disk", geom1=disk_pc, save=False)

        cloud = ContourExtractor()
        cloud.load_cloud(pc_name="circle.ply", parent_folder=save_folder)
        cloud.downsample(voxel_size=0.01)
        cloud.pca_projection()
        cloud.extract(concavity=1.0, length_threshold=0.2)

    
