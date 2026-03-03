# Root Tracking Tool: Readme

* 🇬🇧 English version (default)
* 🇫🇷 [Version française](../fr/README.fr.md)


### Table of Contents

- [Root Tracking Tool: Readme](#root-tracking-tool-readme)
    - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Key features](#key-features)
  - [Compatibility](#compatibility)
  - [Installation](#installation)
    - [Option A: minimal installation (recommended)](#option-a-minimal-installation-recommended)
      - [With PyQt6 and Python 3.10 or higher (recommended)](#with-pyqt6-and-python-310-or-higher-recommended)
      - [With PyQt6 and Python 3.9](#with-pyqt6-and-python-39)
      - [With PyQt5](#with-pyqt5)
    - [Option B: Locked environment (reproductible)](#option-b-locked-environment-reproductible)
  - [Quick start](#quick-start)
    - [Program launch](#program-launch)
  - [User Guide](#user-guide)
  - [Data structure](#data-structure)
    - [Program structure](#program-structure)
    - [Expected input structure](#expected-input-structure)
    - [Output data structure](#output-data-structure)
  - [Technical documentation](#technical-documentation)
  - [Roadmap](#roadmap)
    - [1. Manual segmentation correction tools](#1-manual-segmentation-correction-tools)
    - [2. Performance optimization](#2-performance-optimization)
    - [3. Automated segmentation using deep learning](#3-automated-segmentation-using-deep-learning)



## Description

Root Tracking Tool (RTT) is software dedicated to the segmentation and analysis of root and leaf systems of plants grown in rhizoboxes.

The program processes datasets consisting of pre-segmented images (e.g., via Ilastik), in which roots, leaves, and background are represented by distinct color codes.


## Key features

- Segmentation and cleaning of root/leaf masks
- Construction of the skeleton and root graph
- Automatic extraction of the main trunk
- Complete morphometric analysis
- Interactive visualization
- Structured export of results


## Compatibility

RTT is compatible with:

* Official support: Python 3.10 to 3.14 (PyQt5 or PyQt6)
* Extended compatibility: Python 3.9 (see Installation section, PyQt5 or PyQt6 ≤ 6.6.1 recommended)


## Installation

Two installation methods are available:

* Option A (recommended): minimal installation
* Option B: locked environment (reproducible)

⚠️ It is strongly discouraged to install PyQt5 and PyQt6 simultaneously in the same virtual environment, as this may prevent the program from starting.

The main development was carried out using Python 3.14, which is the recommended version.



### Option A: minimal installation (recommended)

This method installs only the necessary dependencies.

**Important :**

* Under Python 3.9, PyQt6 must be limited to version 6.6.1. Use the file “requirements/minimal-py39-qt6.txt”.
* The *imagecodecs* package is required to open certain images (particularly TIFF). Even if it is not imported directly into the code, its absence can cause a crash when loading images.



#### With PyQt6 and Python 3.10 or higher (recommended)



**Windows**

```
python -m venv .venv
.venv\\Scripts\\activate

pip install -U pip
pip install -r requirements/minimal-qt6.txt
```



**Linux/macOS**

```
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements/minimal-qt6.txt
```



#### With PyQt6 and Python 3.9


**Windows**

```
python -m venv .venv
.venv\\Scripts\\activate

pip install -U pip
pip install -r requirements/minimal-py39-qt6.txt
```


**Linux/macOS**

```
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements/minimal-py39-qt6.txt
```



#### With PyQt5


**Windows**

```
python -m venv .venv
.venv\\Scripts\\activate

pip install -U pip
pip install -r requirements/minimal-qt5.txt
```


**Linux/macOS**

```
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements/minimal-qt5.txt
```


### Option B: Locked environment (reproductible)

Cette méthode permet de reproduire exactement un environnement de test (versions figées des dépendances).


**Example (Python 3.14 + PyQt6) :**

```
pip install -r requirements/lock-py314-qt6.txt
```

Other files are available in the requirements/ folder.

The naming convention is as follows:

lock-py<version>-qt<5|6>.txt


**Example :**

lock-py313-qt6.txt

corresponds to the Python 3.13 test environment with PyQt6.

⚠️ It is recommended to use the Python 3.14 environment for which the main development was carried out.


## Quick start

### Program launch

Activation of the virtual environment:


**Windows**

```
.venv\\Scripts\\activate
```


**Linux/macOS**

```
source .venv/bin/activate
```

Then launch:

```
python main.py
```


## User Guide

Here is a quick user guide. More details can be found in [user_guide.md](user_guide.md).

**Part I : Segmentation**
1. Select the dataset folder
2. Select the output folder
3. Select the datasets to be processed
4. Select the areas of interest (roots and leaves)
5. Start cropping the images
6. Segment the roots and leaves
7. View the results in the “Graph” tabs

**Part II: Root Architecture Analysis**
1. Open the root architecture analysis window.
2. Select the datasets to be processed.
3. Configure the segmentation and analysis options.
4. Start analyzing the current dataset or all selected datasets.
5. View the results using the “Results,” “Graphs,” and “Heatmap” tabs.


## Data structure

The following sections detail the internal organization of the project and data. For standard users, only the section “Expected input structure” is necessary.

### Program structure

```text
RTT/
│
├── Analysis/                 # Default output folder
├── Data/                     # Default input data folder
├── icons/                    # Folder containing the program icons
│   ├── IconRTT.png           # Icon used in the taskbar
│   ├── logo_rtt.png          # Logo used for the PyQt5 version of the program
│   ├── logo_rtt_2.png        # Logo used for the PyQt6 version of the program
│   ├── logo_start.png        # Icon currently unused
│
├── requirements/             # Folder containing the “requirements.txt” files for creating a ‘minimal’ or ‘locked’ environment allowing the program to run
│   ├── lock-py39-qt5.txt     # Requirements file for creating a locked environment with Python 3.9 and PyQt5
│   ├── lock-py39-qt6.txt     # Requirements file for creating a locked environment with Python 3.9 and PyQt6
│   ├── lock-py310-qt5.txt    # Requirements file for creating a locked environment with Python 3.10 and PyQt5
│   ├── lock-py310-qt6.txt    # Requirements file for creating a locked environment with Python 3.10 and PyQt6
│   ├── lock-py311-qt5.txt    # Requirements file for creating a locked environment with Python 3.11 and PyQt5
│   ├── lock-py311-qt6.txt    # Requirements file for creating a locked environment with Python 3.11 and PyQt6
│   ├── lock-py312-qt5.txt    # Requirements file for creating a locked environment with Python 3.12 and PyQt5
│   ├── lock-py312-qt6.txt    # Requirements file for creating a locked environment with Python 3.12 and PyQt6
│   ├── lock-py313-qt5.txt    # Requirements file for creating a locked environment with Python 3.13 and PyQt5
│   ├── lock-py313-qt6.txt    # Requirements file for creating a locked environment with Python 3.13 and PyQt6
│   ├── lock-py314-qt5.txt    # Requirements file for creating a locked environment with Python 3.14 and PyQt5
│   ├── lock-py314-qt6.txt    # Requirements file for creating a locked environment with Python 3.14 and PyQt6
│   ├── minimal-py39-qt6.txt  # Requirements file for creating a minimal environment with Python 3.9 and PyQt6
│   ├── minimal-qt5.txt       # Requirements file for creating a minimal environment with Python ≥ 3.9 and PyQt5
│   ├── minimal-qt6.txt       # Requirements file for creating a minimal environment with Python ≥ 3.10 and PyQt6
│
├── main.py                   # Main file from which to start the program
├── utils.py                  # Functions and utility classes common to other parts of the program
├── widgets.py                # Custom PyQt widgets used by other parts of the program
├── window_analyzer.py        # Window management for advanced root system analysis
```


### Expected input structure

The *Data* folder is the default folder for receiving input datasets. Each dataset must be contained in a folder (for example, *Dataset1* and *Dataset2* in the diagram below). 

```text
RTT/
│
├── Analysis/
├── Data/
│   ├── Dataset1/
│   │   ├── DS1-Image-J01.png
│   │   ├── DS1-Image-J05.png
│   │   ├── DS1-Image-J09.png
│   │   ├── DS1-Image-J13.png
│   │   ├── DS1-Image-J17.png
│   │   ├── DS1-Image-J20.png
│   ├── Dataset2/
│   │   ├── DS2-Image-J01.png
│   │   ├── DS2-Image-J03.png
│   │   ├── DS2-Image-J07.png
│   │   ├── DS2-Image-J12.png
│   │   ├── DS2-Image-J14.png
│   │   ├── DS2-Image-J18.png
│   │   ├── DS2-Image-J20.png
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


### Output data structure

The data exported by the program will be placed in the selected output folder (by default, this is the *Analysis* folder). Let's assume that we have selected the *Data* folder as the input, which itself contains our two folders, *Dataset1* and *Dataset2*. The data obtained during the analyses will then be organized according to the following tree structure:


```text
RTT/
│
├── Analysis/
│   ├── Dataset1/               # Output file containing root and leaf analysis results from *Dataset1*
│   │   ├── Leaves/             # Leaf analysis results
│   │   │   ├── ConvexHull/     # File containing the masks of the convex hull of each image
│   │   │   ├── Crop/           # Folder containing images cropped around the selection of leaves
│   │   │   ├── Results/        # File for exporting results in CSV format
│   │   │   ├── Segmented/      # Folder containing leaf segmentation masks
│   │   │
│   │   ├── Roots/              # Root analysis results
│   │   │   ├── ConvexHull/     # File containing the masks of the convex hull of each image
│   │   │   ├── Crop/           # Folder containing images cropped around the selection of roots
│   │   │   ├── Results/        # File for exporting results in CSV format
│   │   │   ├── Segmented/      # Folder containing root segmentation masks
│   │   │   ├── Skeletonized/   # File containing the root system skeleton and obtained directly from segmentation masks
│   │   │
│   ├── Dataset2/
│   │   ├── Leaves/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │
│   │   ├── Roots/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │   ├── Skeletonized/
│   │   │
│   ├── ...
├── Data/
│   ├── Dataset1/
│   │   ├── ...
│   ├── Dataset2/
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

## Technical documentation

Technical details are available in the following files:

- 📘 [Algorithm description](algorithm.md)
- ⚙ [Parameters reference](parameters.md)
- 📊 [Outputs description](outputs.md)
- 📖 [User guide](user_guide.md)


## Roadmap

The areas of development currently being considered for RTT include:


### 1\. Manual segmentation correction tools


* Polygonal selection of areas
* Brush/pencil tool
* Pipette tool (class selection)
* Fill tool
* Interactive mask editing

**Purpose:** to enable fine correction of segmentation errors prior to analysis.


### 2\. Performance optimization

* Partial rewriting of critical sections in C++ (via pybind11)
* Optimization of graph operations
* Reduction in processing time for large data sets

**Objective:** to improve scalability on large image series.


### 3\. Automated segmentation using deep learning

* Integration of a CNN model for root segmentation
* Option to train on annotated datasets
* Complete pipeline: raw image → segmentation → analysis

**Objective:** to make RTT independent of external segmentation software.



