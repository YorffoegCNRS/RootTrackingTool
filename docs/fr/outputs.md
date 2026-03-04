# RTT : outputs

* 🇫🇷 Version française
* 🇬🇧 [English version](../en/outputs.md)


## 1. Format attendu des noms d'images

RTT exporte des mesures structurées autour de variables communes permettant de relier chaque observation à son contexte : Dataset, Image name, Modality, Day. Ces données sont directement extraites des noms de fichiers qui doivent donc suivre un format particulier pour que le programme puisse les extraire correctement. Le nom du jeu de données est lui directement tiré du nom du dossier parent de chaque jeu d'images.

```text
Format attendu :

<Modality>_<anything>_J<index>.<extension>

Exemples valides :
RGB_root_J12.png
IR_scan_j03.tif
VIS_image_D07.jpg
DS_01_input_d20.tiff
```



## 2. Résultats de l'analyse rapide des racines et des feuilles


La première partie du programme, dont le rôle est la segmentation des images et la préparation des masques permettant une analyse avancée du système racinaire dans sa seconde partie, permet quand même une analyse racinaire et/ou foliaire rapide. Après avoir sélectionné les dossiers à traiter, le dossier de sortie, les zones d'intérêt et segmenter les racines/feuilles, un fichier CSV sera automatiquement exporté pour chaque dataset analysé. Un fichier CSV supplémentaire regroupant l'ensemble des résultats sera également créé dans le cas où la sélection de plusieurs datasets pour l'analyse seraient effectués.

### 1. Analyse d'un seul dataset

Supposons que nous travaillons sur un dataset nommé DS_01, que le dossier de sortie est le dossier par défaut Analysis, alors nous aurons l'arborescence de fichiers suivante :

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

Les dossiers **Leaves** et **Roots** ainsi que leurs sous-dossiers sont automatiquement créés lors de l'analyse. C'est lors de la segmentation des racines et des feuilles que les fichiers CSV **DS_01_roots_analysis.csv** et **DS_01_leaves_analysis.csv** comportant respectivement les résultats des analyses rapides des racines et des feuilles seront créés.


### 2. Analyse de plusieus datasets

Nous supposons dans cette partie qu'au moins 2 datasets DS\_01 et DS\_02 ont été sélectionnés avant de démarrer l'analyse, nous avons dans ce cas l'organisation suivante des fichiers exportés, avec le regroupement des résultats de l'analyse racinaire dans le fichier **global_roots_analysis.csv**, et celui de l'analyse foliaire dans le fichier **global_leaves_analysis**, ces 2 fichier se trouvant directement à la racine du dossier de sortie (ici le dossier **Analysis**). Voici le schéma de l'organisation des fichiers exportés dans cette configuration :


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
│   ├── DS_02/
│   │   ├── Leaves/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │   ├── DS_02_leaves_analysis.csv
│   │   │
│   │   ├── Roots/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   ├── Segmented/
│   │   │   ├── Skeletonized/
│   │   │   ├── DS_02_roots_analysis.csv
├── ...
├── global_leaves_analysis
├── global_roots_analysis.csv
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


### 3. Liste des variables

* Dataset : nom du jeu de données
* Image name : nom de l'image
* Analysis type : type d'analyse (racinaire ou foliaire)
* Pixel count : nombre de pixel représentant les racines / feuilles
* Convex area : nombre de pixels composant l'enveloppe convexe
* Modality : modalité du jeu de données
* Day : index du jour


## 3. Résultats de l'analyse du système racinaire

Deux modes d’export sont possibles :

- Analyse d’un seul dataset → export manuel
- Analyse de plusieurs datasets → export automatique dans DS/Roots/Results

### 1. Analyse du dataset en cours

Le schéma ci-dessous présente l'organisation des fichiers exportés lors de l'analyse du dataset courant (un seul dataset) :

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

Le dossier **DS_01/Roots/Results** contient le fichier CSV **DS_01_root_architecture_results.csv** des résultats de l'analyse du système racinaire, variables affichées dans l'onglet **Results** (et **Heatmap** pour la longueur racinaire par cellule), et le dossier **Visualizations** contenant les images de visualisation du graphe racinaire pour chaque jour de l'expérience. 


### 2. Analyse des datasets séléctionnés

L'organisation des fichiers exportés lors de l'analyse de plusieurs datasets diffère de l'analyse d'un seul dataset. Les fichiers exportés sont les mêmes que pour l'analyse d'un seul dataset, cependant un fichier CSV regroupant l'ensemble des résultats est également créé. Voici donc l'organisation des fichiers exportés dans ce mode :


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
│   ├── DS_02/
│   │   ├── Leaves/
│   │   │   ├── ...
│   │   │
│   │   ├── Roots/
│   │   │   ├── ConvexHull/
│   │   │   ├── Crop/
│   │   │   ├── Results/
│   │   │   │   ├── Visualizations/
│   │   │   │   │   ├── DS_02_root_arch_J005.png
│   │   │   │   │   ├── DS_02_root_arch_J007.png
│   │   │   │   │   ├── DS_02_root_arch_J011.png
│   │   │   │   │   ├── ...
│   │   │   │   │   ├── DS_02_root_arch_J020.png
│   │   │   │   ├── DS_02_root_architecture_results.csv
│   │   │   ├── Segmented/
│   │   │   ├── Skeletonized/
│   │   │   ├── DS_02_roots_analysis.csv
├── ...
├── batch_root_architecture_results.csv
├── global_leaves_analysis
├── global_roots_analysis.csv
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



## 4. Métriques exportées

Les variables sont listées par ordre alphabétique pour faciliter la recherche rapide.
Les variables suffixées `_raw` correspondent aux valeurs avant élagage des petites branches.
Les variables suffixées `_cum` correspondent aux valeurs cumulées temporellement.

| Variable | Type | Unité | Description | Remarques |
|----------|------|-------|------------|-----------|
| branch_count | int | - | Nombre de branches après suppression de la racine principale | Calculé sur le graphe résiduel |
| centroid_x | float | px | Coordonnée x du barycentre | Basé sur le graphe complet |
| centroid_y | float | px | Coordonnée y du barycentre | Basé sur le graphe complet |
| centroid_x_display | float | px | Coordonnée x du barycentre après redimensionnement pour visualisation | Redimensionnement de facteur 1/scale |
| centroid_y_display | float | px | Coordonnée y du barycentre après redimensionnement pour visualisation | Redimensionnement de facteur 1/scale |
| convex_area | float | px² | Surface totale de l'enveloppe convexe du graphe complet | |
| Cx,y | float | px | Longueur totale des racines contenues dans la cellule de la grille virtuelle de X lignes et Y colonnes | Visualisation par heatmap |
| endpoint_count | int | - | Nombre de endpoints (extrémités) après élagage | |
| endpoint_count_raw | int | - | Nombre de endpoints (extrémités) avant élagage | |
| exact_skeleton_length | float | px | Longueur exacte du squelette (métrique 8-connexe pondérée) | Voir section ci-dessous |
| main_root_length | float | px | Longueur de la racine principale | |
| mean_secondary_angles | float | ° | Moyenne des valeurs des angles mesurés entre la direction locale de la racine principale et de la direction initiale de chaque racine secondaire | Angles en degré (°) |
| mean_abs_secondary_angles | float | ° | Moyenne des valeurs absolues des angles mesurés entre la direction locale de la racine principale et de la direction initiale de chaque racine secondaire | Angles en degré (°) |
| root_count | int | - | Nombre de racines secondaires après élagage | = endpoint_count - 1 |
| root_count_cum | int | - | Identique à root_count mais valeur cumulée dans le temps (ne peut jamais diminuer entre deux jours successifs) | Evite la perte de racines |
| root_count_attach | int | - | Nombre de racines secondaires partant de la racine principale | |
| root_count_attach_cum | int | - | Identique à root_count_attach mais valeur cumulée dans le temps (ne peut jamais diminuer entre deux jours successifs) | Evite la perte de racines |
| root_count_raw | int | - | Nombre de racines secondaires avant élagage | = endpoint_count_raw - 1 |
| root_count_raw_cum | int | - | Identique à root_count_raw mais valeur cumulée dans le temps (ne peut jamais diminuer entre deux jours successifs) | Evite la perte de racines |
| scale | float | - | Facteur de redimensionnement de l'image pour la visualisation | Redimensionnement de facteur 1/scale |
| secondary_root_length | float | px | Longueur cumulée des racines secondaires | |
| std_secondary_angles | float | ° | Ecart-type des valeurs des angles mesurés entre la direction locale de la racine principale et de la direction initiale de chaque racine secondaire | Angles en degré (°) |
| std_abs_secondary_angles | float | ° | Ecart-type des valeurs absolues des angles mesurés entre la direction locale de la racine principale et de la direction initiale de chaque racine secondaire | Angles en degré (°) |
| total_area | float | px² | Surface totale du système racinaire | |
| total_root_length | float | px | Longueur totale approximée du graphe | Peut sous-estimer |


Si le facteur d'échelle du nombre de pixels par centimètre a été renseigné avant de lancer l'analyse, alors certaines variables seront "doublées" mais avec le suffixe `_cm` pour indiquer que la valeur est donnée cette fois en centimètres (ou centimètre carré pour les surfaces) et non en pixels. Ce suffixe est potentiellement présent sur les variables suivantes :
* convex_area
* Cx,y
* exact_skeleton_length
* main_root_length
* secondary_root_length
* total_area
* total_root_length


### Notes importantes

- Les longueurs sont calculées sur un squelette binaire 8-connexe.
- Les angles sont exprimés en degrés.
- Les métriques suffixées `_cm` ne sont présentes que si le paramètre pixels/cm est défini.
- Les variables centroid_*_display dépendent du redimensionnement visuel.


### Différence entre `total_root_length` et `exact_skeleton_length`

* **total_root_length**  
  Longueur calculée en parcourant les arêtes du graphe racinaire construit à partir du squelette.  
  Les distances sont évaluées entre nœuds du graphe (distance euclidienne).  
  Le sous-échantillonnage éventuel du graphe (réduction du nombre de nœuds pour optimiser les performances) peut entraîner une légère sous-estimation de la longueur totale.

* **exact_skeleton_length**  
  Longueur calculée directement à partir de tous les pixels du squelette binaire.  
  Pour chaque pixel :
  - distance = 1 pour un voisin horizontal/vertical
  - distance = √2 pour un voisin diagonal  
  Cette méthode correspond à une métrique 8-connexe pondérée permettant une approximation fidèle de la longueur euclidienne en espace discret.

Les deux mesures sont exprimées en pixels. 
Si le paramètre pixels/cm est défini, des colonnes supplémentaires suffixées `_cm` sont ajoutées, elles remplacent les pixels par des centimètres et les pixels² par des cm².