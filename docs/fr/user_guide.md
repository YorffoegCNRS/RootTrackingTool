# RTT : User guide

* 🇫🇷 Version française
* 🇬🇧 [English version](../en/user_guide.md)


## Guide d'utilisation

Le fonctionnement du programme peut être décomposé en deux grandes étapes :

I. Prétraitement et analyse rapide

* Sélection des zones d’intérêt
* Chargement des jeux de données segmentés
* Vérification visuelle des masques
* Analyse rapide des composantes connexes


II. Analyse avancée du réseau racinaire

* Construction du graphe racinaire
* Extraction du squelette
* Calcul des métriques
  * Longueur totale
  * Longueur du squelette exact
  * Ramifications
  * Architecture du réseau
* Export des résultats


### Partie 1 : Prétraitement et analyse rapide


**Etape 1 : Chargement des données**

1. Cliquer sur le bouton ***Select datasets folder***.
2. Sélectionner le dossier contenant les dossiers de vos datasets.
3. Cliquer sur le bouton ***Select output folder***.
4. Sélectionner le dossier dans lequel seront exportées les masques de segmentation et les résultats d'analyse.
5. Dans la liste des jeux de données, sélectionner/désélectionner les datasets voulus pour la segmentation et l'analyse


**Etape 2 : Sélection des zones d'intérêts**

1. Cliquer sur le bouton ***Selection***.
2. Sélectionner sur l'image la partie englobant l'ensemble des racines des jeux de données sélectionnés
3. Cliquer sur le bouton ***Confirm Roots*** pour valider la sélection des racines.
4. Sélectionner sur l'image la partie englobant l'ensemble de la surface foliaire des jeux de données sélectionnés.
5. Cliquer sur le bouton ***Confirm Leaves*** pour valider la sélection des feuilles.


**Etape 3 : Cropping et segmentation**

1. Cliquer sur le bouton ***Crop Roots*** pour cropper les images autour des racines et exporter le résultat.
2. Cliquer sur le bouton ***Crop Leaves*** pour cropper les images autour des feuilles et exporter le résultat.
3. Cliquer sur le bouton ***Segment Roots*** pour ouvrir la fenêtre de paramètres de la segmentation racinaire puis cliquer sur ***Apply*** lorsque le résultat vous convient.
4. Cliquer sur le bouton ***Segment Leaves*** pour ouvrir la fenêtre de paramètres de la segmentation foliaire puis cliquer sur ***Apply*** lorsque le résultat vous convient.
5. Visualisation possible des courbes d'évolution de la taille des feuilles/racines et de leur enveloppe convexe via les onglets ***Roots Graph*** et ***Leaves Graph***.


### Partie 2 : Analyse avancée du réseau racinaire

Si vous avez déjà effectué la partie sur vos jeux de données, vous pouvez passer à la partie 2 simplement après avoir suivant l'étape 1 de la partie 1.

**Etape 1 : Analyse avancée du système racinaire**

1. Cliquer le bouton ***Roots architecture analysis***, une nouvelle fenêtre s'ouvre alors.
2. Si vous avez déjà importé vos jeux de données dans la partie 1, vos datasets devraient être listés dans le menu de gauche. Dans ce cas sélectionnez ceux que vous souhaitez analyser et passer au point 3. Sinon vous pouvez choisir des masques de segmentation que vous avez vous-même créé en cliquant sur le bouton ***Select masks***.
3. Paramétrez les différentes variables pour optimiser au mieux la détection du graphe racinaire, une description détaillée du rôle de chaque paramètre est présente dans le fichier ***parameters.md***.
4. Démarrer l'analyse, soit par le bouton ***Analyze current dataset*** si vous voulez seulement analyser le dataset en cours, soit via le bouton ***Analyze selected datasets*** pour analyser l'ensemble des jeux de données sélectionnés avec les paramètres choisis.


**Etape 2 : Visualisation et export des résultats**

1. Si vous avez lancé l'analyse pour le dataset en cours seulement alors vous devrez exporter les résultats via le bouton ***Export results (CSV)*** pour sauvegarder l'ensemble des mesures au format CSV, et exporter les visualisations en cliquant sur le bouton ***Export visualizations*** puis en choisissant le dossier de destination. Par contre si vous avez choisi d'effectuer l'analyse sur tous les datasets sélectionnés, alors les résultats et les visualisations seront exportées dans les dossiers de sortie des datasets, dans les sous-dossiers Roots\\Results (voir le schéma de la structure des données en sortie pour plus de détails).
2. Une fois l'analyse terminée, il est possible de passer d'un jeu de données à l'autre en cliquant dessus dans le menu de gauche. Il est possible de voir l'évolution du graphe racinaire d'un jeu de données via l'onglet ***Visualization*** en faisant défiler la barre des jours ***Day*** en haut.
3. Les résultats exportés au format CSV peuvent être visualisés via l'onglet ***Results***.
4. La représentation graphique de l'évolution des différentes variables peut être affichée via l'onglet ***Graphs***.

5. L'évolution de la longueur des racines dans chaque sous-section de l'image (en X lignes et Y colonnes choisies avant le démarrage de l'analyse avancée) peut être affichée dans l'onglet ***Heatmap***. Quelques paramètres de visualisation y sont présents : unité, codes couleurs, inversion des couleurs, changement des jours.
