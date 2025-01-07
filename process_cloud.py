# -*- coding:utf-8 -*-

__projet__ = "Point_cloud"
__nom_fichier__ = "process_cloud"
__author__ = "Yanis Sid-Ali Haouchine"
__date__ = "novembre 2024"

import pyvista as pv


def load_cloud(filepath):
    cloud = pv.read(filepath)
    return cloud


def visualize_cloud(pts):
    cloud = pv.PolyData(pts)
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, point_size=5, render_points_as_spheres=False)
    plotter.set_background("grey", top="white")  # Background color
    plotter.enable_eye_dome_lighting()  # Eye Dome Lighting
    #plotter.render_window.SetMultiSamples(8)  # Anti-aliasing
    #plotter.enable_point_picking(callback=lambda p: print(f"Selected point : {p}"))
    plotter.show()


point_cloud = load_cloud("cave_res_1cm_cross_section.ply")
visualize_cloud(point_cloud)
