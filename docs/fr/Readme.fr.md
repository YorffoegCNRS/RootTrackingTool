# Root Tracking Tool


### Table des matières

- [Root Tracking Tool](#root-tracking-tool)
    - [Table des matières](#table-des-matières)
  - [Description](#description)
  - [Fonctionnalités principales](#fonctionnalités-principales)
  - [Compatibilité](#compatibilité)
  - [Installation](#installation)
    - [Option A : installation rapide (Recommandé)](#option-a--installation-rapide-recommandé)
      - [Avec PyQt6 et Python 3.10 ou supérieur (Recommandé)](#avec-pyqt6-et-python-310-ou-supérieur-recommandé)
      - [Avec PyQt6 et Python 3.9](#avec-pyqt6-et-python-39)
      - [Avec PyQt5](#avec-pyqt5)
    - [Option B : Environnement verrouillé (Reproductible)](#option-b--environnement-verrouillé-reproductible)
  - [Lancement rapide (Quick start)](#lancement-rapide-quick-start)
    - [Lancement du programme](#lancement-du-programme)
  - [Guide d'utilisation](#guide-dutilisation)
  - [Structure des données](#structure-des-données)
    - [Structure du programme](#structure-du-programme)
    - [Structure attendue en entrée](#structure-attendue-en-entrée)
    - [Structure des données en sortie](#structure-des-données-en-sortie)
  - [Documentation technique](#documentation-technique)
  - [Roadmap](#roadmap)
    - [1. Outils de correction manuelle de segmentation](#1-outils-de-correction-manuelle-de-segmentation)
    - [2. Optimisation des performances](#2-optimisation-des-performances)
    - [3. Segmentation automatisée par apprentissage profond](#3-segmentation-automatisée-par-apprentissage-profond)



## Description

Root Tracking Tool (RTT) est un logiciel dédié à la segmentation et à l’analyse des systèmes racinaires et foliaires de plantes cultivées en rhizoboxes.

Le programme traite des jeux de données constitués d’images pré-segmentées (par exemple via Ilastik), dans lesquelles les racines, les feuilles et l’arrière-plan sont représentés par des codes couleur distincts.


## Fonctionnalités principales

- Segmentation et nettoyage de masques racinaires/foliaires
- Construction du squelette et du graphe racinaire
- Extraction automatique du tronc principal
- Analyse morphométrique complète
- Visualisation interactive
- Export structuré des résultats


## Compatibilité


RTT est compatible avec :

* Support officiel : Python 3.10 à 3.14 (PyQt5 ou PyQt6)
* Compatibilité étendue : Python 3.9 (voir section Installation, PyQt5 ou PyQt6 ≤ 6.6.1 recommandé)


## Installation

Deux méthodes d’installation sont proposées :

* Option A (recommandée) : installation minimale
* Option B : environnement verrouillé (reproductible)

⚠️ Il est fortement déconseillé d’installer PyQt5 et PyQt6 simultanément dans le même environnement virtuel, cela peut empêcher le programme de démarrer.

Le développement principal a été réalisé sous Python 3.14, qui est la version recommandée.



### Option A : installation rapide (Recommandé)

Cette méthode installe uniquement les dépendances nécessaires.

**Important :**

* Sous Python 3.9, PyQt6 doit être limité à la version 6.6.1. Utilisez le fichier "requirements/minimal-py39-qt6.txt".
* Le package *imagecodecs* est requis pour l’ouverture de certaines images (notamment TIFF). Même s’il n’est pas importé directement dans le code, son absence peut provoquer un crash lors du chargement des images.



#### Avec PyQt6 et Python 3.10 ou supérieur (Recommandé)



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



#### Avec PyQt6 et Python 3.9


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



#### Avec PyQt5


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


### Option B : Environnement verrouillé (Reproductible)

Cette méthode permet de reproduire exactement un environnement de test (versions figées des dépendances).


**Exemple (Python 3.14 + PyQt6) :**

```
pip install -r requirements/lock-py314-qt6.txt
```

D’autres fichiers sont disponibles dans le dossier requirements/.

La nomenclature est la suivante :

lock-py<version>-qt<5|6>.txt


**Exemple :**

lock-py313-qt6.txt

correspond à l'environnement de test Python 3.13 avec PyQt6.

⚠️ Il est recommandé d’utiliser l’environnement Python 3.14 pour lequel le développement principal a été effectué.


## Lancement rapide (Quick start)

### Lancement du programme

Activation de l’environnement virtuel :


**Windows**

```
.venv\\Scripts\\activate
```

**Linux/macOS**

```
source .venv/bin/activate
```


Puis lancer :

```
python main.py
```


## Guide d'utilisation

Voici un guide d'utilisation rapide. Plus de détails sont présents dans [user_guide.md](user_guide.md).

**Partie I : Segmentation**
1. Sélectionner le dossier des datasets
2. Sélectionner le dossier de sortie
3. Sélectionner les datasets à traiter
4. Sélectionner les zones d'intérêts (racines et feuilles)
5. Lancer le cropping des images
6. Segmenter les racines et les feuilles
7. Visualisation des résultats dans les onglets "Graph"

**Partie II : Analyse du système racinaire**
1. Ouvrir la fenêtre d'analyse de l'architecture racinaire
2. Choisir les datasets à traiter
3. Paramétrer les options de segmentation et d'analyse
4. Démarrer l'analyse du dataset en cours ou de tous les datasets sélectionnés
5. Visualisation des résultats via les onglets "Results", "Graphs" et "Heatmap"


## Structure des données

Les sections suivantes détaillent l’organisation interne du projet et des données. Pour les utilisateurs standards, seule la section “Structure attendue en entrée” est nécessaire.

### Structure du programme

```text
RTT/
│
├── Analysis/                 # Dossier de sortie par défaut
├── Data/                     # Dossier des données d'entrée par défaut
├── icons/                    # Dossier contenant les icônes du programme
│   ├── IconRTT.png           # Icône utilisé dans la barre des tâches
│   ├── logo_rtt.png          # Logo utilisé pour la version PyQt5 du programme
│   ├── logo_rtt_2.png        # Logo utilisé pour la version PyQt6 du programme
│
├── requirements/             # Dossier contenant les fichiers "requirements.txt" pour la création d'environnement 'minimal' ou 'verrouillé' permettant l'exécution du programme
│   ├── lock-py39-qt5.txt     # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.9 et PyQt5
│   ├── lock-py39-qt6.txt     # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.9 et PyQt6
│   ├── lock-py310-qt5.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.10 et PyQt5
│   ├── lock-py310-qt6.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.10 et PyQt6
│   ├── lock-py311-qt5.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.11 et PyQt5
│   ├── lock-py311-qt6.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.11 et PyQt6
│   ├── lock-py312-qt5.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.12 et PyQt5
│   ├── lock-py312-qt6.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.12 et PyQt6
│   ├── lock-py313-qt5.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.13 et PyQt5
│   ├── lock-py313-qt6.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.13 et PyQt6
│   ├── lock-py314-qt5.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.14 et PyQt5
│   ├── lock-py314-qt6.txt    # Fichier 'requirements' pour la création d'un environnement verrouillé avec Python 3.14 et PyQt6
│   ├── minimal-py39-qt6.txt  # Fichier 'requirements' pour la création d'un environnement minimal avec PyQt6
│   ├── minimal-qt5.txt       # Fichier 'requirements' pour la création d'un environnement minimal avec PyQt5
│   ├── minimal-qt6.txt       # Fichier 'requirements' pour la création d'un environnement minimal avec PyQt6
│
├── main.py                   # Fichier principal à partir duquel démarrer le programme
├── utils.py                  # Fonctions et classes utilitaires communes aux autres parties du programme
├── widgets.py                # Widgets PyQt personnalisés et utilisés par les autres parties du programme
├── window_analyzer.py        # Gestion de la fenêtre permettant l'analyse avancée du système racinaire
```


### Structure attendue en entrée

Le dossier *Data* est le dossier choisi par défaut pour recevoir les jeux de données en entrée. Chaque jeu de données doit être contenu dans un dossier (par exemple *Dataset1* et *Dataset2* dans le schéma ci-dessous). 

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


### Structure des données en sortie

Les données exportées par le programme seront placées dans le dossier de sortie choisi (par défaut c'est le dossier *Analysis*). Supposons que nous avons sélectionné le dossier *Data* en entrée, contenant lui-même nos 2 dossiers *Dataset1* et *Dataset2*. Alors les données obtenus lors des analyses seront organisées selon l'arborescence suivante :


```text
RTT/
│
├── Analysis/
│   ├── Dataset1/               # Dossier de sortie contenant les résultats d'analyse racinaire et foliaire de *Dataset1*
│   │   ├── Leaves/             # Résultats de l'analyse foliaire
│   │   │   ├── ConvexHull/     # Dossier contenant les masques de l'enveloppe convexe de chaque image
│   │   │   ├── Crop/           # Dossier contenant les images croppées autour de la sélection des feuilles
│   │   │   ├── Results/        # Dossier destiné à l'export des résultats au format CSV
│   │   │   ├── Segmented/      # Dossier contenant les masques de segmentation des feuilles
│   │   │
│   │   ├── Roots/              # Résultats de l'analyse racinaire
│   │   │   ├── ConvexHull/     # Dossier contenant les masques de l'enveloppe convexe de chaque image
│   │   │   ├── Crop/           # Dossier contenant les images croppées autour de la sélection des racines
│   │   │   ├── Results/        # Dossier destiné à l'export des résultats au format CSV
│   │   │   ├── Segmented/      # Dossier contenant les masques de segmentation des racines
│   │   │   ├── Skeletonized/   # Dossier contenant le squelette du système racinaire et obtenu directement à partir des masques de segmentation
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

## Documentation technique

Les détails techniques sont disponibles dans les fichiers suivants :

- 📘 [Algorithm description](algorithm.md)
- ⚙ [Parameters reference](parameters.md)
- 📊 [Outputs description](outputs.md)
- 📖 [User guide](user_guide.md)


## Roadmap

Les axes de développement actuellement envisagés pour RTT incluent :


### 1\. Outils de correction manuelle de segmentation

* Sélection polygonale de zones
* Outil pinceau / crayon
* Outil pipette (sélection de classe)
* Outil de remplissage
* Modification interactive des masques

**Objectif :** permettre une correction fine des erreurs de segmentation avant analyse.


### 2\. Optimisation des performances

* Réécriture partielle des sections critiques en C++ (via pybind11)
* Optimisation des opérations sur les graphes
* Réduction du temps de traitement des grands jeux de données

**Objectif :** améliorer la scalabilité sur des séries d'images volumineuses


### 3\. Segmentation automatisée par apprentissage profond

* Intégration d’un modèle CNN pour la segmentation des racines
* Possibilité d’entraînement sur jeux annotés
* Pipeline complet : image brute → segmentation → analyse

**Objectif :** rendre RTT indépendant d'un logiciel externe de segmentation

