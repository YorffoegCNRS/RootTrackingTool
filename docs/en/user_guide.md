# RTT : User guide

* 🇬🇧 English version (default)
* 🇫🇷 [Version française](../fr/user_guide.md)


## User guide

The program's operation can be broken down into two main stages:

I. Preprocessing and quick analysis

* Selection of regions of interest
* Loading segmented datasets
* Visual inspection of the masks
* Quick analysis of connected components


II. Advanced root architecture analysis

* Construction of the root graph
* Extraction of the skeleton
* Metrics calculation
  * Total length
  * Exact skeleton length
  * Branches
  * Network architecture
* Export of results


### Part 1 : Preprocessing and quick analysis

**Step 1 : Loading data**

1. Click on the ***Select datasets folder*** button.
2. Select the directory containing your dataset folders.
3. Click on the ***Select output folder*** button.
4. Select the directory to which the segmentation masks and analysis results will be exported.
5. In the list of datasets, select/deselect the desired datasets for segmentation and analysis.


**Step 2 : Select regions of interest**

1. Click on the ***Selection*** button.
2. Select the part of the image that encompasses all the roots of the selected data sets.
3. Click the ***Confirm Roots*** button to confirm the root selection.
4. Select the part of the image that encompasses the entire leaf surface of the selected datasets.
5. Click on the ***Confirm Leaves*** button to confirm the leaves selection.


**Step 3 : Cropping and segmentation**

1. Click on the ***Crop Roots*** button to crop the images around the roots and export the result.
2. Click on the ***Crop Leaves*** button to crop the images around the leaves and export the result.
3. Click the ***Segment Roots*** button to open the root segmentation settings window, then click ***Apply*** when you are satisfied with the result.
4. Click the ***Segment Leaves*** button to open the leaves segmentation settings window, then click ***Apply*** when you are satisfied with the result.
5. You can view graphs showing the evolution of leaf/root size and their convex envelope via the ***Roots Graph*** and ***Leaves Graph*** tabs.


### Part 2 : Advanced root achitecture analysis

If you have already completed Part 1 on your datasets, you can proceed to Part 2 immediately after completing Step 1 of Part 1.


**Step 1 : Advanced root achitecture analysis**

1. Click on the ***Roots architecture analysis*** button, a new window will then open.
2. If you have already imported your datasets in Part 1, they should be listed in the left-hand menu. In this case, select the ones you want to analyze and proceed to step 3. Otherwise, you can choose segmentation masks that you have created yourself by clicking on the ***Select masks button***.
3. Set the different variables to optimize root graph detection. A detailed description of the role of each parameter is provided in the ***parameters.md*** file.
4. Start the analysis either by clicking the ***Analyze current dataset*** button if you only want to analyze the current dataset, or by clicking the ***Analyze selected datasets*** button to analyze all selected datasets with the chosen parameters.


**Step 2 : Visualization and axporting results**

1. If you have run the analysis for the current dataset only, you will need to export the results using the ***Export results (CSV)*** button to save all measurements in CSV format, and export the visualizations by clicking on the ***Export visualizations*** button and then choosing the destination folder. However, if you have chosen to perform the analysis on all selected datasets, the results and visualizations will be exported to the dataset output folders, in the Roots\\Results subfolders (see the output data structure diagram for more details).
2. Once the analysis is complete, you can switch between datasets by clicking on them in the left-hand menu. You can view the evolution of a dataset's root graph via the ***Visualization*** tab by scrolling through the ***Day*** bar at the top.
3. Results exported in CSV format can be viewed via the ***Results*** tab.
4. A graphical representation of the evolution of the different variables can be displayed via the ***Graphs*** tab.
5. The evolution of root length in each subsection of the image (in X rows and Y columns chosen before starting the advanced analysis) can be displayed in the ***Heatmap*** tab. Several visualization parameters are available: unit, color codes, color inversion, and day change.

