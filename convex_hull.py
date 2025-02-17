import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import alphashape

# Importing the cloud
point_cloud_name = "filtered_cross_section_clean.ply"
point_cloud = o3d.io.read_point_cloud("saved_clouds/" + point_cloud_name)

# Reducing the cloud
point_cloud = point_cloud.voxel_down_sample(voxel_size=4.0)
points = np.asarray(point_cloud.points)

# Verify the number of points
if points.shape[0] < 3:
    raise ValueError("Not enough points to create a contour.")

# Project the cloud in the 2D Y-Z plan
points_2d = points[:, 1:3]

# Calculate the alpha shape
alpha = 0.2
alpha_shape = alphashape.alphashape(points_2d, alpha)

# Displaying in 3D
plt.figure(figsize=(8, 6))
plt.scatter(points_2d[:, 0], points_2d[:, 1], c='blue', label="Projected points")

# Displaying the alpha shape
if alpha_shape:
    x, y = alpha_shape.exterior.xy
    plt.plot(x, y, 'r-', label="Alpha-shape")

plt.legend()
plt.title("2D Projection & Alpha-shape")
plt.xlabel("Y")
plt.ylabel("Z")
plt.axis("equal")
plt.show()
