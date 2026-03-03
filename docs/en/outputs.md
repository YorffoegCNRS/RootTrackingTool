# RTT : outputs

* 🇬🇧 English version (default)
* 🇫🇷 [Version française](docs/fr/outputs.md)


## 1. Expected format for image names

RTT exports structured measurements based on common variables that link each observation to its context: Dataset, Image name, Modality, Day. This data is extracted directly from the file names, which must therefore follow a specific format so that the program can extract them correctly. The name of the dataset is taken directly from the name of the parent folder of each set of images.

```text
Expected format :

<Modality>_<anything>_J<index>.<extension>

Valid examples :
RGB_root_J12.png
IR_scan_j03.tif
VIS_image_D07.jpg
DS_01_input_d20.tiff
```

## 2. Results of the quick analysis of roots and leaves

The first part of the program, which segments images and prepares masks for advanced root system analysis in the second part, still allows for rapid root and/or leaf analysis. After selecting the files to be processed, the output folder, the areas of interest, and segmenting the roots/leaves, a CSV file will be automatically exported. Let's assume that we are working on a dataset named DS_01, and that the output folder is the default Analysis folder. We will then have the following file tree:

```text
RTT\
│
├── Analysis/
│   ├── DS_01/
│   │   ├── Leaves/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │   ├── DS_01_leaves_analysis.csv
│   │   │
│   │   ├── Roots/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │   ├── Skeletonized/
│   │   │   ├── DS_01_roots_analysis.csv
│ 
├── ...
│
├── Data/
│   ├── Dataset1/
│   │   ├── ...
│   ├── ...
├── icons/
│   ├── IconRTT.png
│   ├── logo_rtt.png
│   ├── logo_rtt_2.png
│
├── main.py
├── utils.py
├── widgets.py
├── window_analyzer.py
```

The **Leaves** and **Roots** folders and their subfolders are automatically created during analysis. During the segmentation of roots and leaves, the CSV files **DS_01_roots_analysis.csv** and **DS_01_leaves_analysis.csv** will be created, containing the results of the rapid analysis of roots and leaves, respectively.

### Variables list

* Dataset : dataset name
* Image name : image name
* Analysis type : analysis type roots or leaves
* Pixel count : number of pixels composing the roots/leaves
* Convex area : number of pixels composing the convex hull
* Modality : modality of the dataset
* Day : day index


## 3. Results of the quick roots analysis

Two export modes are available:

- Analysis of a single dataset → manual export
- Analysis of multiple datasets → automatic export to DS/Roots/Results


```text
RTT\
│
├── Analysis/
│   ├── DS_01/
│   │   ├── Leaves/
│   │   │   ├── ...
│   │   │
│   │   ├── Roots/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   │   ├── Visualizations/
│   │   │   │   │   ├── DS_01_root_arch_J006.png
│   │   │   │   │   ├── DS_01_root_arch_J007.png
│   │   │   │   │   ├── DS_01_root_arch_J010.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── DS_01_root_arch_J020.png
│   │   │   │   ├── DS_01_root_architecture_results.csv
│   │   │   ├── Segmented/
│   │   │   ├── Skeletonized/
│   │   │   ├── DS_01_roots_analysis.csv
│ 
├── ...
│
├── Data/
│   ├── ...
├── icons/
│   ├── ...
├── main.py
├── utils.py
├── widgets.py
├── window_analyzer.py
```

The **DS_01/Roots/Results** folder contains the CSV file **DS_01_root_architecture_results.csv** with the results of the root system analysis, variables displayed in the ***Results*** tab (and ***Heatmap*** for root length per cell), and the ***Visualizations*** containing visualization images of the root graph for each day of the experiment.


### Important notes

- Lengths are calculated on an 8-connected binary skeleton.
- Angles are expressed in degrees.
- Metrics suffixed with `_cm` are only present if the pixels/cm parameter is defined.
- The centroid_*_display variables depend on visual resizing.


## Exported metrics

The variables are listed in alphabetical order to facilitate quick searching.
Variables suffixed with `_raw` correspond to values before pruning small branches.
Variables suffixed with `_cum` correspond to values accumulated over time.

| Variable | Type | Unity | Description | Comments |
|----------|------|-------|------------|-----------|
| branch_count | int | - | Number of branches after removal of the main root | Calculated on the residual graph |
| centroid_x | float | px | x-coordinate of the centroid | Based on the complete graph |
| centroid_y | float | px | y-coordinate of the centroid | Based on the complete graph |
| centroid_x_display | float | px | x-coordinate of the centroid after resizing for visualization | Resizing by a factor of 1/scale |
| centroid_y_display | float | px | y-coordinate of the centroid after resizing for visualization | Resizing by a factor of 1/scale |
| convex_area | float | px² | Total area of the convex hull of the complete graph | |
| Cx,y | float | px | Total length of roots contained in the virtual grid cell of X rows and Y columns | Heatmap visualization |
| endpoint_count | int | - | Number of endpoints after pruning | |
| endpoint_count_raw | int | - | Number of endpoints before pruning | |
| exact_skeleton_length | float | px | Exact skeleton length (weighted 8-connected metric) | See section below |
| main_root_length | float | px | Length of the main root | |
| mean_secondary_angles | float | ° | Mean of the values of the angles measured between the local direction of the main root and the initial direction of each secondary root | Angles in degrees (°) |
| mean_abs_secondary_angles | float | ° | Mean of the absolutes values of the angles measured between the local direction of the main root and the initial direction of each secondary root | Angles in degrees (°) |
| root_count | int | - | Number of secondary roots after pruning | = endpoint_count - 1 |
| root_count_cum | int | - | Identical to root_count but cumulative value over time (can never decrease between two consecutive days) | Prevents root loss |
| root_count_attach | int | - | Number of secondary roots extending from the main root | |
| root_count_attach_cum | int | - | Identical to root_count_attach but cumulative value over time (can never decrease between two consecutive days) | Prevents root loss |
| root_count_raw | int | - | Number of secondary roots before pruning | = endpoint_count_raw - 1 |
| root_count_raw_cum | int | - | Identical to root_count_raw but cumulative value over time (can never decrease between two consecutive days) | Prevents root loss |
| scale | float | - | Image resizing factor for visualization | Resizing by a factor of 1/scale |
| secondary_root_length | float | px | Cumulative length of secondary roots | |
| std_secondary_angles | float | ° | Standard deviation of the values of the angles measured between the local direction of the main root and the initial direction of each secondary root | Angles in degrees (°) |
| std_abs_secondary_angles | float | ° | Standard deviation of the values of the angles measured between the local direction of the main root and the initial direction of each secondary root | Angles in degrees (°) |
| total_area | float | px² | Total root system surface area | |
| total_root_length | float | px | Approximate total length of the graph | May underestimate |


If the scale factor for the number of pixels per centimeter was entered before running the analysis, then certain variables will be “doubled” but with the suffix `_cm` to indicate that the value is now given in centimeters (or square centimeters for areas) rather than pixels. This suffix may appear on the following variables:
* convex_area
* Cx,y
* exact_skeleton_length
* main_root_length
* secondary_root_length
* total_area
* total_root_length

### Difference between `total_root_length` and `exact_skeleton_length`

* **total_root_length**  
  Length calculated by traversing the edges of the root graph constructed from the skeleton.
  Distances are evaluated between nodes in the graph (Euclidean distance).
  Any subsampling of the graph (reducing the number of nodes to optimize performance) may result in a slight underestimation of the total length.

* **exact_skeleton_length**  
  Length calculated directly from all pixels in the binary skeleton.
  For each pixel:
  - distance = 1 for a horizontal/vertical neighbor
  - distance = √2 for a diagonal neighbor
  This method corresponds to a weighted 8-connected metric that provides an accurate approximation of Euclidean length in discrete space.

Both measurements are expressed in pixels.
If the pixels/cm parameter is defined, additional columns suffixed with `_cm` are added, replacing pixels with centimeters and pixels² with cm².