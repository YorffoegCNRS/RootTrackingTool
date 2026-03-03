# RTT : Parameters

## 1. Paramètres – Segmentation (Partie I)


### RGB Threshold
- **Variables associées** : `red_range, green_range, blue_range, red_invert, green_invert, blue_invert`
- **Type** : `list[int, int], list[int, int], list[int, int], bool, bool, bool`
- **Unité** : `-`
- **Description** :
  - `red_range, green_range, blue_range : intervalles de valeurs de pixels choisies`
  - `red_invert, green_invert, blue_invert : si désactivé on garde les pixels dont la valeur appartient à l'intervalle correspondant, sinon on garde les pixels dont la valeur n'appartient pas à l'intervalle correspondant`
- **Effet** :
  - `Si intervalle trop restrictif → suppression trop importante de l'objet à segmenter`
  - `Si intervalle trop permissif → trop d'objets indésirables et de bruits vont rester lors de la segmentation`
- **Valeurs conseillées** : `cela dépend du code couleur utilisée pour représenter les racines/feuilles. Si les feuilles sont vertes le mieux est d'augmenter la valeur minimale du vert et de ne pas inverser l'intervalle, ainsi tous les objets qui auront une composante "green" trop basse disparaîtront de l'image.`

---

### Fusion previous masks
- **Variable associée** : `fusion_masks`
- **Type** : `bool`
- **Unité** : `-`
- **Description** : `Active la fusion temporelle des masks (opération **OR** entre chaque masque et le masque précédent).`
- **Effet** :
  - `Activé → permet potentiellement de faire apparaître des racines effacées au jour J qui seraient visibles au jour J-1. Si les masques ne sont pas parfaitement alignés, cette option est dangereuse (dédoublement des racines).`
  - `Désactivé → pas d'effet`
- **Valeur conseillée** : `Désactivée sauf si vous savez vraiment pourquoi vous l'utilisez. Une version moins brutale existe et peut être utilisée en partie II.`


### Keep max connected component only
- **Variable associée** : `keep_max_component`
- **Type** : `bool`
- **Unité** : `-`
- **Description** : `Active/désactive la suppression de tout ce qui est extérieur au plus grand objet.`
- **Effet** :
  - `Activé → Supprime tous les objets extérieurs au plus grand, peut supprimer potentiellement des objets appartenant à aux racines / feuilles`
  - `Désactivé → Aucun effet`
- **Valeur conseillée** : `actif seulement sur la segmentation des feuilles, si la feuille est bien représentée d'un seul bloc`

---

### Minimum connected component area
- **Variable associée** : `min_connected_components_area`
- **Type** : `int`
- **Unité** : `pixels²`
- **Description** : `Supprime toutes les composantes connexes dont l'aire est inférieure à ce seuil. Certaines composantes supprimées peuvent être rétablies par le critère suivant (maximum centroid distance).`
- **Effet** :
  - `Trop petit → bruit résiduel`
  - `Trop grand → suppression de racines fines`
- **Valeur conseillée** : `200 à 600 suivant la résolution des images et de la valeur du paramètre suivant`


### Maximum centroid distance
- **Variable associée** : `max_centroid_dst`
- **Type** : `float`
- **Unité** : `pixels`
- **Description** : `Agit comme un filtrage des composantes supprimées par **min_connected_components_area**, les composantes dont le centroide est à une distance inférieure à ce seuil du reste des objets restants ne sont pas supprimées.`
- **Effet** :
  - `Trop petit : Aucune des composantes supprimées ne sera rétablies, ce paramètre sera donc inutile`
  - `Trop grand : Toutes les composantes supprimées seront rétablies, donc aucun effet de ce paramètre et du précédent, en plus d'alourdir les calculs`
- **Valeur conseillée** : `entre 50.0 et 200.0 suivant la résolution de l'image et de la répartition des objets.`

---

### Minimum object size
- **Variable associée** : `minimum_object_size`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Supprime tous les objets dont la taille est inférieure à cette valeur`
- **Effet** :
  - `Trop petit → bruit résiduel`
  - `Trop grand → suppression de racines fines`
- **Valeur conseillée** : `50–300 selon résolution`


---

### Closing kernel size
- **Variable associée** : `kernel_size`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Rayon du noyau utilisé pour la fermeture morphologique.`
- **Effet** :
  - `Trop petit → trous non comblés`  
  - `Trop grand → fusion excessive`
- **Valeur conseillée** : `entre 3 et 9 en fonction de la résolution`

### Closing kernel shape
- **Variable associée** : `kernel_shape`
- **Type** : `int (index de la forme : 0 rectangle, 1 croix, 2 ellipse)`
- **Unité** : `-`
- **Description** : `Influence la façon d'agencer les pixels autour du noyau en fonction de sa forme.`
- **Effet** : `L'effet dépend vraiment du contexte, de l'agencement des trous que l'on cherche à combler.`
- **Valeur conseillée** : `en général c'est le rectangle qui donne de meilleurs résultats.`


## 2. Paramètres – Analyse du système racinaire (Partie II)

### Maximum image size
- **Variable associée** : `max_image_size`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Taille maximale de la plus grande dimension (hauteur ou largeur) de l'image au-dessus de laquelle on redimensionne l'image à une taille inférieure (rescale = 2000 / max(width, height) ratio conservé pour les 2 dimensions de l'image).`
- **Effet** : 
  - `Valeur trop faible → l'image sera fortement redimensionnée ce qui entraînera potentiellement une perte de précision`
  - `Valeur trop forte → une image de grande taille risque de ne pas être redimensionnée et entraîner une augmentation significative du temps de calcul` 
- `Diminue la taille de l'image si besoin pour réduire les temps de calcul. Cela peut également entraîner une perte de précision dans le calcul des métriques (généralement négligeable)`
- **Valeur conseillée** : `2000 pixels maximum semble être une valeur raisonnable à affiner en fonction des capacités du PC que vous utilisez.`

### Maximum pixel before sampling
- **Variable associée** : `min_sampling_threshold`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Nombre maximum de pixels avant d'activer le sous-échantillonnage des points du graphe.`
- **Effet** : 
  - `Valeur trop faible → l'échantillonnage va être effectué sur de petits graphes et entraîner une perte de précision alors que le PC était capable de traiter le graphe en un temps raisonnable`
  - `Valeur trop importante → l'échantillonnage risque de ne jamais se déclencher ce qui entraînera une explosion du temps de calcul si le PC n'est pas assez puissant`
- **Valeur conseillée** : `Cela dépend des capacités de votre PC. Vous pouvez commencé par tenter d'analyser un jeu de données, et si les résultats sont trop longs à arriver, vous pouvez arrêter l'analyse avec le bouton **Stop** et diminuer ce seuil.`

### Sampling
- **Variable associée** : `connection_sample_rate`
- **Type** : `float`
- **Unité** : `-`
- **Description** : `Proportion des points du graphe gardés si l'échantillonnage est activé.`
- **Effet** :
  - `Valeur = 1.0 → 100% les points sont gardés, échantillonnage désactivé`
  - `Valeur < 1.0 → échantillonnage activé si le nombre de point du graphe ≥ min_sampling_threshold`
- **Valeur conseillée** : `dépend de la taille du graphe et de la configuration du PC. Si le PC est assez puissant vous pouvez le désactiver via la valeur 1.0, sinon descendre cette valeur jusqu'à obtenir un temps d'exécution raisonnable.`

### Maximum iterations
- **Variable associée** : `max_connect_iterations`
- **Type** : `int`
- **Unité** : `-`
- **Description** : `Nombre maximum d'itérations (de boucles) exécutées par l'algorithme pour tenter de connecter les objets.`
- **Effet** :
  - `Valeur trop faible → exécution rapide mais échec de la reconnexion de nombreux objets`
  - `Valeur trop forte → exécution lente, augmentation du nombre d'objets connectés`
- **Valeur conseillée** : `Autour de 10, réduire légèrement ce nombre si le temps d'exécution devient trop long.`

---

### Closing radius
- **Variable associée** : `closing_radius`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Rayon du noyau utilisé pour la fermeture morphologique.`
- **Effet** :
  - `Trop petit → trous non comblés`
  - `Trop grand → fusion excessive`
- **Valeur conseillée** : `entre 3 et 11`

### Minimum branch size
- **Variable associée** : `min_branch_length`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Taille minimum en pixels d'une racine pour être comptabilisée comme telle.`
- **Effet** :
  - `Valeur trop faible → des départs de quelques pixels risquent d'être comptabilisés comme des racines`
  - `Valeur trop forte → de nombreuses racines ne seront pas comptabilisés car de taille inférieure à ce nombre`
- **Valeur conseillée** : `entre 20 et 50 suivant la résolution de l'image.`

### Minimum object size
- **Variable associée** : `min_object_size`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Taille minimum d'un objet sous lequel il est retiré de l'image.`
- **Effet** :
  - `Valeur trop faible → ce paramètre n'aura que très peu, voire aucun **Effet**`
  - `Valeur trop forte → des objets appartenant au système racinaire peuvent être supprimés`
- **Valeur conseillée** : `entre 50 et 300 suivant la résolution de l'image et la qualité de la segmentation.`

### Maximum connection distance
- **Variable associée** : `max_connection_dst`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Distance maximale entre 2 objets à reconnecter.`
- **Effet** :
  - `Valeur trop faible → peu d'objets seront connectés`
  - `Valeur trop forte → risque de connexion entre objets non désirés `
- **Valeur conseillée** : `dépend fortement de la résolution des images et de la taille des fractures présentes dans les racines.`

### Connection thickness
- **Variable associée** : `line_thickness`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Si la connexion entre objets est activée (paramètre **connect_objects**), cette variable définit l'épaisseur de la ligne connectant les 2 objets.`
- **Effet** :
  - `Valeur trop faible → connexion trop fine`
  - `Valeur trop forte → la connexion effectuée va être trop épaisse et peu potentiellement rejoindre d'autres racines`
- **Valeur conseillée** : `Cela dépend de l'épaisseur des racines présentes sur vos images, la valeur de 5 est un bon compromis dans notre cas.`

### Main path bias
- **Variable associée** : `main_path_bias`
- **Type** : `int`
- **Unité** : `-`
- **Note** : `La valeur est utilisée comme facteur multiplicatif dans le calcul des poids du graphe.`
- **Description** : `Valeur du biais de continuité sur les arêtes du graphe et ainsi contraindre la racine principale à suivre le même chemin.`
- **Effet** : 
  - `Valeur trop faible → le poids risque d'être lui aussi trop faible et le tracé de la racine principale risque de dévier`
  - `Valeur trop forte → peut empêcher l'algorithme de trouver la bonne prolongation`
- **Valeur conseillée** : `20`

### Fusion tolerance pixel
- **Variable associée** : `fusion_tolerance_pixel`
- **Type** : `int`
- **Unité** : `pixels`
- **Description** : `Taille du rayon de fermeture appliquée au masque avant intersection avec le masque du jour précédent (si le paramètre temporal_merge est activé).`
- **Effet** :
  - `Valeur trop faible → risque de n'avoir que peu ou pas d'effet`
  - `Valeur trop forte → risque de faire apparaître un dédoublement des racines si les masques sont mal alignés`
- **Valeur conseillée** : `De 3 à 11 en fonction de la résolution de l'image`

### Pixels/cm
- **Variable associée** : `pixels_per_cm`
- **Type** : `float`
- **Unité** : `pixels/cm`
- **Description** : `Permet de convertir certaines métriques données en pixels ou pixels², en mesures physiques réelles, en centimètres.`
- **Effet** : 
  - `Valeur nulle : pas de conversion en mesures physiques réelles.`
  - `Valeur positive : ajoute de nouvelles variables en sortie, ces variables sont simplement la conversion de certaines mesures en centimètres.`
- **Valeur conseillée** : `Si vous connaissez ce facteur de conversion il est fortement conseillé de l'indiquer.`

---

### Temporal fusion
- **Variable associée** : `temporal_merge`
- **Type** : `bool`
- **Unité** : `-`
- **Description** : `Active la fusion temporelle locale, c'est-à-dire pour chaque masque au jour J, on applique une fermeture morphologique de rayon **fusion_tolerance_pixel** et on intersecte ce masque avec celui du jour J-1.`
- **Effet** : `Permet de faire réapparaître des morceaux manquants de racines visibles au jour précédent.`
- **Valeur conseillée** : `Cette méthode de fusion étant bien plus douce que l'algorithme de fusion en partie I, il est donc plutôt conseillé d'utiliser celui-ci et de désactiver celui de la partie I.`

### Connect objects
- **Variable associée** : `connect_objects`
- **Type** : `bool`
- **Unité** : `-`
- **Description** : `Active la tentative de reconstruction des racines coupées.`
- **Effet** : 
  - `Activé → peut combler les fractures potentiels dans les racines, mais cela peut potentiellement reconnecter des points n'appartenant pas une même racine`
  - `Désactivé → les fractures présentes dans les racines ne seront pas comblés`
- **Valeur conseillée** : `Si des racines comportent des petites zones manquantes, il est conseillé d'activer ce paramètre tout en ajustant bien **max_connection_dst**.`

---

### Grid rows
- **Variable associée** : `grid_rows`
- **Type** : `int`
- **Unité** : `-`
- **Description** : `Indique le nombre de lignes la grille virtuelle partageant l'image en **grid_rows** lignes et **grid_cols** colonnes.`
- **Effet** : `Plus ce nombre est grand, plus le nombre de cellule composant la grille sera important.`
- **Valeur conseillée** : `Cela dépend vraiment de vos besoins en termes de métriques/variables décrivant le système racinaire.`

### Grid columns
- **Variable associée** : `grid_cols`
- **Type** : `int`
- **Unité** : `-`
- **Description** : `Indique le nombre de colonnes la grille virtuelle partageant l'image en **grid_rows** lignes et **grid_cols** colonnes.`
- **Effet** : `Plus ce nombre est grand, plus le nombre de cellule composant la grille sera important.`
- **Valeur conseillée** : `Cela dépend vraiment de vos besoins en termes de métriques/variables décrivant le système racinaire.`



## Réglages conseillés (point de départ)

Voici les valeurs que nous utilisons pour l'ensemble des paramètres. Ces valeurs sont données à titre indicatif et vous donnent un point de départ. Elles doivent évidemment être mises en perspective en fonction de la résolution des images analysées, de la qualité de la segmentation, etc.

### Partie I : Segmentation

Valeurs des paramètres utilisées sur des images de taille originale 3528x6228 puis après cropping :
- Roots : 3300x4550
- Leaves : 3200x1600

**Segmentation du système racinaire**
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

**Segmentation des feuilles**
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


### Partie II : Analyse du système racinaire

Analyse effectuée sur des images de taille 3300x4550 une fois découpée.

- max_image_size : 2000 → redimensionnement à 1450x2000 pixels
- min_sampling_threshold : 100000
- connection_sample_rate : 1.0 → échantillonnage désactivé car analyse sur machine puissante
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