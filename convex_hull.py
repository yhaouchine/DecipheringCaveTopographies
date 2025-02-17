import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import alphashape
from scipy.spatial import KDTree

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

# Check if the alpha shape was successfully generated
if alpha_shape is None or not hasattr(alpha_shape, "exterior"):
    raise ValueError("Alpha-shape computation failed, try adjusting alpha.")

x, y = alpha_shape.exterior.xy
contour_2d = np.column_stack((x, y))

# Build a KDTree to find the closest original 3D points corresponding to the 2D contour
tree = KDTree(points_2d)
_, indices_3d = tree.query(contour_2d)

# Retrieve the full 3D coordinates (X, Y, Z) of the contour
contour_3d = points[indices_3d]

# === 5. PLOT THE 3D CONTOUR USING MATPLOTLIB ===
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the original point cloud
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1, label="Point cloud")

# Plot the reconstructed 3D contour
ax.plot(contour_3d[:, 0], contour_3d[:, 1], contour_3d[:, 2], 'r-', linewidth=2, label="Alpha-shape 3D")

ax.set_title("3D Point Cloud & Alpha-shape Contour")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.axis("equal")  # Ensure equal axis scaling
ax.legend()
plt.show()