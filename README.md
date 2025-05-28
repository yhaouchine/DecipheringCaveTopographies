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

1. Place your 3D point cloud data (e.g., `.ply`, `.las`, `.xyz`) in the `data/` directory.
2. Run the main script to process the data:

    ```bash
    python reconstruct.py --input data/your_cave_file.ply --output results/
    ```

3. View the generated 2D reconstructions in the `results/` directory.

## Dependencies

- Python 3.8+
- numpy
- open3d
- matplotlib
- scipy

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Example

![Example 2D Reconstruction](docs/example_output.png)

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.