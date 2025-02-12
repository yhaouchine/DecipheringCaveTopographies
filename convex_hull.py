import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, KDTree

def angle_between_vectors(v1, v2):
    """Retourne l'angle entre deux vecteurs en radians."""
    v1 = v1.astype(np.float64)  # Convertir en float64
    v2 = v2.astype(np.float64)  # Convertir en float64
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


# Charger le nuage de points
point_cloud_name = "filtered_cross_section_clean.ply"
point_cloud = o3d.io.read_point_cloud("saved_clouds/" + point_cloud_name)

# Réduction du nuage de points
point_cloud = point_cloud.voxel_down_sample(voxel_size=4.0)  # Ajuste le paramètre
points = np.asarray(point_cloud.points)

# Vérifier qu'on a assez de points
if points.shape[0] < 3:
    raise ValueError("Pas assez de points pour calculer une enveloppe convexe.")

# Calcul de l'enveloppe convexe
hull = ConvexHull(points)
boundary_points = points[hull.vertices]  # Sélection des points de la coque convexe

# KDTree pour chercher les voisins
tree = KDTree(boundary_points)
neighbors = tree.query(boundary_points, k=4)[1]  # Prend plusieurs voisins pour éviter les erreurs

# Trier les voisins en fonction de l'angle pour assurer un contour continu
ordered_edges = []
visited = set()
current_index = 0  # Départ du premier point de l'enveloppe

for _ in range(len(boundary_points) - 1):
    visited.add(current_index)
    current_point = boundary_points[current_index]

    # Trouver le meilleur voisin basé sur l'angle
    best_index = None
    best_angle = float('inf')
    ref_vector = np.array([1, 0, 0])  # Référence de direction

    for idx in neighbors[current_index]:
        if idx in visited:
            continue

        candidate_vector = boundary_points[idx] - current_point
        angle = angle_between_vectors(ref_vector, candidate_vector)

        if angle < best_angle:
            best_angle = angle
            best_index = idx

    if best_index is not None:
        ordered_edges.append([current_index, best_index])
        current_index = best_index

# Création du LineSet pour Open3D
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(boundary_points)
line_set.lines = o3d.utility.Vector2iVector(np.array(ordered_edges, dtype=np.int32))
line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(ordered_edges))  # Rouge

# Affichage
o3d.visualization.draw_geometries([point_cloud, line_set])
