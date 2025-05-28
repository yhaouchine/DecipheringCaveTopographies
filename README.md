# Two-Dimensional Reconstruction of Cave Geometry from 3D Point Clouds

## Overview

This project provides tools and methods for reconstructing two-dimensional section of caves from three-dimensional point cloud data. 

## Features

- Import and preprocess 3D point cloud data (PLY files): Define the section, clean noisy points.
- Convert 3D section point cloud to 2D.
- Compute the contour of the section point cloud.
- Interpolate points in sparse areas.
- Export results in VTK or CSV.

## Installation

```bash
git clone https://github.com/yhaouchine/DecipheringCaveTopographies.git
cd DecipheringCaveTopographies
pip install -r requirements.txt
```

## Usage
Documentation is available here: https://yhaouchine.github.io/DecipheringCaveTopographies/

0. (miscellaneous) polygones_test.py is used to create simple polygones (Sphere, Cuboid, Pyramid) to test the workflow.
1. Run process_cloud.py to import the cave point cloud, define the section to extract and save the extracted section as a PLY file.
2. Run clean_cloud.py to import the previously saved section cloud, and perform and ellispoidal cleaning of the point cloud.
3. Run contour_extractor to import the cleaned section cloud and compute its contour.

## Dependencies

- Python 3.9+

Install all dependencies with:

```bash
pip install -r requirements.txt
```