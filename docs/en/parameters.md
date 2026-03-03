# RTT : Parameters

* 🇬🇧 English version (default)
* 🇫🇷 [Version française](../fr/parameters.md)


## 1. Parameters – Segmentation (Part I)

### RGB Threshold
- **Related variables** : `red_range, green_range, blue_range, red_invert, green_invert, blue_invert`
- **Type** : `list[int, int], list[int, int], list[int, int], bool, bool, bool`
- **Unity** : `-`
- **Description** :
  - `red_range, green_range, blue_range : intervals of selected pixel values`
  - `red_invert, green_invert, blue_invert : If disabled, pixels whose value belongs to the corresponding interval are retained; otherwise, pixels whose value does not belong to the corresponding interval are retained.`
- **Effect** :
  - `If interval too restrictive → excessive removal of the object to be segmented`
  - `If the interval is too permissive → too many unwanted objects and noise will remain during segmentation.`
- **Recommended values** : `It depends on the color code used to represent the roots/leaves. If the leaves are green, it is best to increase the minimum value of green and not reverse the interval, so that all objects with a “green” component that is too low will disappear from the image.`

---

### Fusion previous masks
- **Related variable** : `fusion_masks`
- **Type** : `bool`
- **Unity** : `-`
- **Description** : `Enables temporal fusion of masks (**OR** operation between each mask and the previous mask).`
- **Effect** :
  - `Enabled → potentially allows roots that were deleted on day D to appear, which would have been visible on day D-1. If the masks are not perfectly aligned, this option is dangerous (duplication of roots).`
  - `Disabled → no effect`
- **Recommended value** : `Disabled unless you really know why you are using it. A less aggressive version exists and can be used in part II.`


### Keep max connected component only
- **Related variable** : `keep_max_component`
- **Type** : `bool`
- **Unity** : `-`
- **Description** : `Enable/disable the removal of everything that is external to the largest object.`
- **Effect** :
  - `Enabled → removes all objects outside the largest one, may potentially remove objects belonging to roots/leaves`
  - `Disabled → no effect`
- **Recommended value** : `active only on leaf segmentation, if the leaf is well represented as a single block`

---

### Minimum connected component area
- **Related variable** : `min_connected_components_area`
- **Type** : `int`
- **Unity** : `pixels²`
- **Description** : `Removes all related components whose area is smaller than this threshold. Some removed components may be restored by the following criterion (maximum centroid distance).`
- **Effect** :
  - `Value too low → residual noise`
  - `Value too high → removal of fine roots`
- **Recommended value** : `200 to 600 depending on the image resolution and the value of the following parameter.`


### Maximum centroid distance
- **Related variable** : `max_centroid_dst`
- **Type** : `float`
- **Unity** : `pixels`
- **Description** : `Acts as a filter for components removed by **min_connected_components_area**; components whose centroid is less than this threshold distance from the rest of the remaining objects are not removed.`
- **Effect** :
  - `Value too low → none of the deleted components will be restored, so this setting will be useless.`
  - `Value too high → all deleted components will be restored, so there will be no effect from this parameter and the previous one, in addition to slowing down calculations.`
- **Recommended value** : `between 50.0 and 200.0 depending on the image resolution and the distribution of objects.`

---

### Minimum object size
- **Related variable** : `minimum_object_size`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Deletes all objects smaller than this value`
- **Effect** :
  - `Value too low → residual noise`
  - `Value too high → removal of fine roots`
- **Recommended value** : `50–300 depending on the image resolution`


---

### Closing kernel size
- **Related variable** : `kernel_size`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Radius of the nucleus used for morphological closure.`
- **Effect** :
  - `Value too low → unfilled holes`  
  - `Value too high → excessive fusion`
- **Recommended value** : `between 3 and 9 depending on the image resolution`

### Closing kernel shape
- **Related variable** : `kernel_shape`
- **Type** : `int (shape index : 0 rectangle, 1 cross, 2 ellipse)`
- **Unity** : `-`
- **Description** : `Influences how pixels are arranged around the kernel based on its shape.`
- **Effect** : `The effect really depends on the context, on the arrangement of the gaps that we are trying to fill.`
- **Recommended value** : `In general, rectangles give the best results.`


## 2. Parameters – Analysis of the root architecture (Part II)

### Maximum image size
- **Related variable** : `max_image_size`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Maximum size of the largest dimension (height or width) of the image above which the image is resized to a smaller size (rescale = 2000 / max(width, height) ratio maintained for both dimensions of the image).`
- **Effect** : 
  - `Value too low → the image will be significantly resized, potentially resulting in a loss of precision.`
  - `Value too high → a large image may not be resized, resulting in a significant increase in calculation time.` 
- `Reduce the image size if necessary to reduce calculation times. This may also result in a loss of accuracy in the calculation of metrics (usually negligible).`
- **Recommended value** : `A maximum of 2000 pixels seems reasonable, but this should be adjusted according to the capabilities of the PC you are using.`

### Maximum pixel before sampling
- **Related variable** : `min_sampling_threshold`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Maximum number of pixels before activating graph point subsampling.`
- **Effect** : 
  - `Value too low → sampling will be performed on small graphs, resulting in a loss of accuracy, even though the PC was capable of processing the graph in a reasonable amount of time.`
  - `Value too high → sampling may never be triggered, which will cause the calculation time to skyrocket if the PC is not powerful enough.`
- **Recommended value** : `It depends on your PC's capabilities. You can start by trying to analyze a dataset, and if the results take too long to arrive, you can stop the analysis with the **Stop** button and lower this threshold.`

### Sampling
- **Related variable** : `connection_sample_rate`
- **Type** : `float`
- **Unity** : `-`
- **Description** : `Proportion of graph points retained if sampling is enabled.`
- **Effect** :
  - `Value = 1.0 → 100% points retained, sampling disabled`
  - `Value < 1.0 → sampling enabled if the number of points in the graph ≥ min_sampling_threshold`
- **Recommended value** : `depends on the size of the graph and the configuration of the PC. If the PC is powerful enough, you can disable it by setting the value to 1.0. Otherwise, lower this value until you obtain a reasonable execution time.`

### Maximum iterations
- **Related variable** : `max_connect_iterations`
- **Type** : `int`
- **Unity** : `-`
- **Description** : `Maximum number of iterations (loops) executed by the algorithm in an attempt to connect objects.`
- **Effect** :
  - `Value too low → fast execution but failure to reconnect many objects`
  - `Value too high → slow execution, increase in the number of connected objects`
- **Recommended value** : `Around 10, reduce this number slightly if the execution time becomes too long.`

---


### Closing radius
- **Related variable** : `closing_radius`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Radius of the nucleus used for morphological closure.`
- **Effect** :
  - `Value too low → unfilled holes`
  - `Value too high → excessive fusion`
- **Recommended value** : `between 3 and 11.`

### Minimum branch size
- **Related variable** : `min_branch_length`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Minimum pixel size of a root to be counted as such.`
- **Effect** :
  - `Value too low → a few pixels may be counted as roots`
  - `Value too high → many roots will not be counted because they are smaller than this number.`
- **Recommended value** : `between 20 and 50 depending on the image resolution.`

### Minimum object size
- **Related variable** : `min_object_size`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Minimum size of an object below which it is removed from the image.`
- **Effect** :
  - `Value too low → this parameter will have very little or no **Effect**.`
  - `Value too high → objects belonging to the root system may be deleted.`
- **Recommended value** : `between 50 and 300 depending on the image resolution and segmentation quality.`

### Maximum connection distance
- **Related variable** : `max_connection_dst`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Maximum distance between two objects to be reconnected.`
- **Effect** :
  - `Value too low → few objects will be connected`
  - `Value too high → risk of connection between unwanted objects`
- **Recommended value** : `depends heavily on the resolution of the images and the size of the fractures present in the roots.`

### Connection thickness
- **Related variable** : `line_thickness`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `If the connection between objects is enabled (**connect_objects** parameter), this variable defines the thickness of the line connecting the two objects.`
- **Effect** :
  - `Value too low → connection too thin`
  - `Value too high → the connection made will be too thick and may potentially join other roots`
- **Recommended value** : `It depends on the thickness of the roots in your images; a value of 5 is a good compromise in our case.`

### Main path bias
- **Related variable** : `main_path_bias`
- **Type** : `int`
- **Unity** : `-`
- **Note** : `The value is used as a multiplier in calculating the weights of the graph.`
- **Description** : `Value of the continuity bias on the edges of the graph, thereby forcing the main root to follow the same path.`
- **Effect** : 
  - `Value too low → the weight may also be too low and the main root may deviate`
  - `Value too high → may prevent the algorithm from finding the correct extension`
- **Recommended value** : `20`

### Fusion tolerance pixel
- **Related variable** : `fusion_tolerance_pixel`
- **Type** : `int`
- **Unity** : `pixels`
- **Description** : `Closure radius size applied to the mask before intersection with the previous day's mask (if the temporal_merge parameter is enabled).`
- **Effect** :
  - `Value too low → risk of having little or no effect`
  - `Value too high → risk of root duplication if masks are misaligned`
- **Recommended value** : `From 3 to 11 depending on the image resolution`

### Pixels/cm
- **Related variable** : `pixels_per_cm`
- **Type** : `float`
- **Unity** : `pixels/cm`
- **Description** : `Converts certain given metrics into pixels or pixels², into actual physical measurements, into centimeters.`
- **Effect** : 
  - `Zero value: no conversion to actual physical measurements.`
  - `Positive value: adds new output variables; these variables are simply the conversion of certain measurements into centimeters.`
- **Recommended value** : `If you know this conversion factor, it is strongly recommended that you indicate it.`

---

### Temporal fusion
- **Related variable** : `temporal_merge`
- **Type** : `bool`
- **Unity** : `-`
- **Description** : `Activates local temporal fusion, i.e., for each mask on day D, a morphological closure with a radius of **fusion_tolerance_pixel** is applied and this mask is intersected with the mask from day D-1.`
- **Effect** : `Allows missing pieces of roots visible the previous day to reappear.`
- **Recommended value** : `Since this merging method is much gentler than the merging algorithm in Part I, it is advisable to use this one and disable the one in Part I.`

### Connect objects
- **Related variable** : `connect_objects`
- **Type** : `bool`
- **Unity** : `-`
- **Description** : `Activates the attempt to rebuild cut roots.`
- **Effect** : 
  - `Enabled → can fill potential gaps in roots, but this may potentially reconnect points that do not belong to the same root`
  - `Disabled → fractures present in the roots will not be filled.`
- **Recommended value** : `If roots have small missing areas, it is advisable to enable this setting while adjusting **max_connection_dst** accordingly.`

---

### Grid rows
- **Related variable** : `grid_rows`
- **Type** : `int`
- **Unity** : `-`
- **Description** : `Indicates the number of rows in the virtual grid dividing the image into **grid_rows** rows and **grid_cols** columns.`
- **Effect** : `The larger this number, the greater the number of cells comprising the grid.`
- **Recommended value** : `It really depends on your needs in terms of metrics/variables describing the root system.`

### Grid columns
- **Related variable** : `grid_cols`
- **Type** : `int`
- **Unity** : `-`
- **Description** : `Indicates the number of columns in the virtual grid dividing the image into **grid_rows** rows and **grid_cols** columns.`
- **Effect** : `The larger this number, the greater the number of cells comprising the grid.`
- **Recommended value** : `It really depends on your needs in terms of metrics/variables describing the root system.`



## Recommended settings (start point)

Here are the values we use for all parameters. These values are provided for informational purposes only and give you a starting point. They must obviously be considered in light of the resolution of the images analyzed, the quality of the segmentation, etc.

### Part I : Segmentation

Parameter values used on images with original size 3528x6228 and after cropping:
- Roots : 3300x4550
- Leaves : 3200x1600

**Root system segmentation**
- red_range : (0, 180)
- red_invert : True
- green_range : (0, 255)
- green_invert : False
- blue_range : (0, 255)
- blue_invert : False
- fusion_masks : False
- keep_max_component : False
- min_connected_components_area : 600
- max_centroid_dst : 200.0
- minimum_object_size : 0
- kernel_size : 3
- kernel_shape : 0 (Rectangle)

**Leaves segmentation**
- red_range : (0, 255)
- red_invert : False
- green_range : (50, 255)
- green_invert : False
- blue_range : (0, 255)
- blue_invert : False
- fusion_masks : False
- keep_max_component : True
- min_connected_components_area : 0
- max_centroid_dst : 0.0
- minimum_object_size : 0
- kernel_size : 5
- kernel_shape : 0 (Rectangle)


### Part II : Root architecture analysis

Analysis performed on images measuring 3300x4550 once cropped.

- max_image_size : 2000 → resizing to 1450x2000 pixels
- min_sampling_threshold : 100000
- connection_sample_rate : 1.0 → sampling disabled because analysis on powerful machine
- max_connect_iterations : 10
- closing_radius : 5
- min_branch_length : 20
- minimum_object_size : 200
- max_connection_dst : 240
- line_thickness : 5
- main_path_bias : 20
- fusion_tolerance_pixel : 5
- pixels_per_cm : 221.0
- temporal_merge : True
- connect_objects : True
- grid_rows : 3

- grid_cols : 2
