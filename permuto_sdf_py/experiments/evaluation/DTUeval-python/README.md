# DTUeval-python

A python implementation of DTU MVS 2014 evaluation. It only takes 1min for each mesh evaluation. And the gap between the two implementations is negligible. 

## Setup and Usage

This script requires the following dependencies.

```
numpy open3d scikit-learn tqdm scipy multiprocessing argparse
```

Download the STL point clouds and Sample Set and prepare the ground truth folder as follows.

```
<dataset_dir>
- Points
    - stl
        - stlxxx_total.ply
- ObsMask
    - ObsMaskxxx_10.mat
    - Planexxx.mat
```

Run the evaluation script (e.g. scan24, mesh mode)
```
python eval.py --data <input> --scan 24 --mode mesh --dataset_dir <dataset_dir> --vis_out_dir <out_dir_for_visualization>
```

## Discussion on randomness
There is randomness in point cloud downsampling in both versions. It iterates through the points and delete the points with distance < 0.2. So the order of points matters. We randomly shuffle the points before downsampling. 

## Comparison with the official script
We evaluate a set of meshes from Colmap and compare the results. We run our script 10 times and take the average. 

|     | diff/official | official | py_avg   | py_std/official |
|-----|---------------|----------|----------|-----------------|
| 24  | 0.0184%       | 0.986317 | 0.986135 | 0.0108%         |
| 37  | 0.0001%       | 2.354124 | 2.354122 | 0.0091%         |
| 40  | 0.0038%       | 0.730464 | 0.730492 | 0.0104%         |
| 55  | 0.0436%       | 0.530899 | 0.531131 | 0.0104%         |
| 63  | 0.0127%       | 1.555828 | 1.556025 | 0.0118%         |
| 65  | 0.0409%       | 1.007686 | 1.008098 | 0.0080%         |
| 69  | 0.0082%       | 0.888434 | 0.888361 | 0.0125%         |
| 83  | 0.0207%       | 1.136882 | 1.137117 | 0.0096%         |
| 97  | 0.0314%       | 0.907528 | 0.907813 | 0.0089%         |
| 105 | 0.0129%       | 1.463337 | 1.463526 | 0.0118%         |
| 106 | 0.1424%       | 0.785527 | 0.786646 | 0.0151%         |
| 110 | 0.0592%       | 1.076125 | 1.075488 | 0.0132%         |
| 114 | 0.0049%       | 0.436169 | 0.436190 | 0.0074%         |
| 118 | 0.1123%       | 0.679574 | 0.680337 | 0.0099%         |
| 122 | 0.0347%       | 0.726771 | 0.726519 | 0.0178%         |
| avg | 0.0153%       | 1.017711 | 1.017867 |                 |


## Error visualization
`vis_xxx_d2s.ply` and `vis_xxx_s2d.ply` are error visualizations.
- Blue: Out of bounding box or ObsMask
- Green: Errors larger than threshold (20)
- White to Red: Errors counted in the reported statistics