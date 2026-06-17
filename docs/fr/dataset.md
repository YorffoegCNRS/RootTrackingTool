# Dataset

## Overview

Le jeu de données associé à RootTrackingTool (RTT) est disponible publiquement via le DOI suivant :

https://doi.org/10.57745/SW015R

Cet ensemble de données a été créé afin d'évaluer les performances de RTT et de le comparer aux logiciels d'analyse d'images de racines existants. Il contient toutes les images et mesures nécessaires pour reproduire les analyses présentées dans la publication associée.


## Contenu du jeu de données

Le dépôt contient :

* Des numérisations brutes de plantes cultivées dans des systèmes « rhizobox » ;
* Des images traitées avec Ilastik et utilisées comme données d'entrée pour RTT ;
* Des numérisations brutes de plantes Arabidopsis thaliana cultivées dans des boîtes de Pétri ;
* Des mesures quantitatives utilisées pour l'évaluation comparative des logiciels ;
* Des mesures de référence réalisées manuellement à l'aide d'ImageJ.


## Espèces de plantes disponibles

Le jeu de données inclut des images de différentes espèces de plantes :

* Arabidopsis thaliana
* Laitue
* Maïs
* Tomate

Les images ont été acquises dans différentes conditions expérimentales et à plusieurs stades de développement.


## Structure des dossiers

```text
Arabidopsis/
├── Images brutes
└── Images traitées avec Ilastik

Lettuce/
├── Images brutes
└── Images traitées avec Ilastik

Maize/
├── Images brutes
└── Images traitées avec Ilastik

Tomato/
├── Images brutes
└── Images traitées avec Ilastik

Raw_data/
└── Software_comparison.xlsx
```


## Données de référence

Le fichier `Software_comparison.xlsx` contient les mesures utilisées pour toutes les analyses comparatives de logiciels présentées dans la publication de RTT.

Il comprend :

* Des mesures manuelles obtenues avec ImageJ ;
* Des mesures du nombre de racines ;
* Des mesures de la longueur totale des racines ;
* Des mesures générées par :

  * RootTrackingTool (RTT)
  * RhizoVision Explorer
  * FarIA
  * RootSystemAnalyzer


## Reproductibilité

Cet ensemble de données fournit toutes les images et mesures nécessaires pour reproduire les analyses comparatives présentées dans la publication RTT.

Il peut également servir de jeu de données de référence pour évaluer et comparer des logiciels d'analyse d'images de racines.


## Citation

Si vous utilisez RTT ou cet ensemble de données dans le cadre de travaux universitaires, veuillez citer la publication RTT correspondante.

DOI du jeu de données :

https://doi.org/10.57745/SW015R

L'article de recherche correspondant est actuellement en cours d'évaluation.
