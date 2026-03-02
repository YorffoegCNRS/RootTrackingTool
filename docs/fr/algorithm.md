# RTT : Algorithm

## Architecture

RTT est organisé en deux modules principaux, correspondant à deux étapes de traitement complémentaires : (i) segmentation/nettoyage des masques issus d’Ilastik, puis (ii) analyse avancée du réseau racinaire par squelettisation et analyse de graphe.


### 1\. Segmentation et préparation des masques (Partie I)


**Entrées :** images pré-segmentées (codes couleur racines / feuilles / fond).

**Sorties :** masques binaires nettoyés et séries d’images exportées (racines et/ou feuilles) + graphiques de suivi.


Le workflow principal est le suivant :

* Sélection du dossier de jeux de données + dossier de sortie
* Définition des zones d'intérêt (racines et feuilles)
* Découpage (cropping) sur les zones d'intérêt
* Segmentation par seuillage RGB, avec options de post-traitement :
  * Conservation de la plus grande composante connexe
  * Filtrage par aire minimale, avec "récupération" des petits objets proches via la distance au centroïde
  * Supression des petits objets (min object size)
  * Fermeture morphologique (taille + forme du noyau)
  * Mise à jour d'une prévisualisation avec application globale
* Export des masques et affichage de graphes d'évolution (taille des racines/feuilles et de l'enveloppe convexe)



### 2\. Analyse du système racinaire (Partie II)

Cette partie transforme le masque binaire racinaire en une représentation topologique (squelette + graphe), puis calcule des métriques morphométriques globales et locales.

Pipeline conceptuel :

```
Masque binaire
      │
      ▼
Nettoyage morphologique
      │
      ▼
Squelettisation
      │
      ▼
Construction du graphe
      │
      ▼
Extraction du chemin principal
      │
      ▼
Extraction des branches secondaires
      │
      ▼
Calcul des métriques
```

#### a. Prétraitement du masque

* Redimensionnement optionnel si l'image dépasse une taille maximale (afin de limiter le coût de calcul)
* Fusion temporelle locale optionnelle (continuité entre jours), avec une fermeture (paramètre *fusion tolerance pixel* pour la taille du noyau) appliquée avant une opération **ET** entre les masques
* Nettoyage morphologique : fermeture (*paramètre closing radius* pour la taille du noyau) + suppression des petits objets (paramètre *minimum object size*)
* Connexion optionnelle de fragments proches (distance max avec le paramètre *maximum connection distance* + épaisseur de connexion via le paramètre *connection thickness*)
* Calibration optionnelle (paramètre *pixels/cm*) : génération de métriques en unités physiques
* Grille virtuelle (X lignes × Y colonnes) : calcul de longueurs racinaires par cellule + visualisation en heatmap


#### b. Squelettisation et construction du graphe

* Extraction d'un squelette morphologique d'épaisseur 1 pixel conservant la topologie
* Construction d'un graphe (8-connexités)
  * Nœuds : pixels du squelette (coordonnées y, x)
  * Arêtes : connexion 8-connexe, pondérée par la distance euclidienne discrète.
  * Poids :
    - voisin horizontal/vertical : 1
    - voisin diagonal : √2
  * Degré d'un nœud : nombre de nœuds voisins
  * Détection des extrémités (nœuds de degré 1) et jonctions (nœuds de degré ≥ 3)


#### c. Détection de la racine principale et analyse temporelle

* Jour 1 : Sélection d'un point de départ (extrémité la plus haute) et d'arrivée (extrémité la plus basse), puis plus court chemin pondéré calculé via l’algorithme de Dijkstra sur graphe non orienté.
* Jours suivants : utilisation du chemin du jour précédent comme référence, via un biais de continuité (paramètre *main path bias*) appliquée aux poids des arêtes, afin de stabiliser la détection dans le temps (voir la partie dédiée ci-dessous pour plus de détails).
* Mécanisme de persistance : si aucun chemin n'est trouvé, le chemin du jour précédent est conservé.
* Extension du chemin principal vers le haut et vers le bas jusqu'à atteindre des extrémités plausibles (processus de ces prolongements ci-dessous).


**Biais de continuité temporelle :**

Afin de stabiliser la détection de la racine principale entre deux jours, un biais est appliqué aux poids des arêtes du graphe. 
* Pour chaque nœud, on calcule la distance euclidienne minimale entre ce nœud et l’ensemble des pixels constituant le chemin principal du jour précédent.
* Pour chaque arête (u, v), on définit : `d̄ = (d(u) + d(v)) / 2`
* Le poids de l’arête devient `w_biaisé = w_base × (1 + α × (d̄ / d_max)^2)` où
  - w_base ∈ {1, √2} (8-connexité),
  - d_max est la distance maximale observée sur l’ensemble des nœuds du graphe,
  - α = 2 × main_path_bias
* Le plus court chemin est ensuite calculé via l’algorithme de Dijkstra sur ces poids biaisés.
* Ce mécanisme introduit une régularisation spatiale continue, limitant les sauts topologiques brutaux entre jours consécutifs.


**Prolongement vers le haut :**

* Depuis le premier point du chemin (ancre haute)
* BFS unidirectionnel : explore tous les nœuds accessibles sans jamais passer par les nœuds déjà dans le chemin principal. Ce BFS est contraint au sous-graphe obtenu après suppression temporaire des arêtes appartenant au chemin principal.
* Parmi toutes les extrémités atteintes, sélectionne celle avec la coordonnée Y minimale (la plus haute)
* Trace le chemin unique depuis l'ancre jusqu'à cette extrémité via reconstruction du dictionnaire parent
* Garantit l'impossibilité de création d'une double branche ou d'un cycle


**Prolongement vers le bas :**

* Depuis le dernier point du chemin (ancre basse)
* BFS unidirectionnel : explore tous les nœuds accessibles sans passer par les nœuds déjà dans le chemin (sauf l'ancre elle-même). Ce BFS est contraint au sous-graphe obtenu après suppression temporaire des arêtes appartenant au chemin principal.
* Parmi toutes les extrémités atteintes, sélectionne celle avec la coordonnée Y maximale (la plus basse)
* Trace le chemin unique depuis l'ancre jusqu'à cet endpoint
* Gère les déviations latérales en fin de racine : même si la racine bifurque à gauche ou à droite en bas, l'algorithme trouve l'extrémité la plus basse atteignable


Cette stratégie garantit que :

* Le chemin principal est toujours simple (pas de boucle, pas de double branche)
* L'extension atteint les vraies extrémités globales même en cas de bifurcations multiples
* Le résultat est reproductible (pas de choix ambigus aux bifurcations)


#### d. Extraction des racines secondaires

Après suppression des nœuds appartenant au chemin principal, chaque composante connexe restante est considérée comme une racine secondaire potentielle.

**Processus :**
1. Suppression du chemin principal du graphe
2. Détection des composantes connexes restantes (NetworkX)
3. Pour chaque composante : calcul du chemin le plus long
4. Élagage optionnel des terminaisons courtes (< min_branch_length)

Les longueurs sont calculées en parcourant le squelette pixel par pixel pour garantir une précision maximale (exact_skeleton_length).


### Complexité algorithmique

Soit N le nombre de pixels du squelette, E le nombre d'arêtes du graphe (E ≈ cN (avec c ≈ 4))

**Construction et prétraitement :**
- Construction du graphe : O(N)
- Détection des endpoints : O(N)
- Calcul des composantes connexes : O(N)

**Détection racine principale :**
- *Avec référence temporelle (jours suivants)* :
  - Calcul des poids biaisés : O(E) = O(N)
  - Dijkstra avec biais : O(E log N) = O(N log N)
- *Sans référence (premier jour ou graphe déconnecté)* :
  - Tri des composantes : O(C log C) où C = nombre de composantes
  - Dijkstra par composante : O(N_i log N_i), N_i = taille composante i

**Extension du chemin (BFS strict) :**
- BFS vers le haut : O(N)
- BFS vers le bas : O(N)
- Reconstruction des chemins : O(L) où L = longueur extension

**Extraction racines secondaires :**
- Détection composantes résiduelles : O(N)
- Calcul longueurs : O(N)

**Complexité totale : O(N log N)** (dominée par Dijkstra mais reste compatible avec des graphes contenant plusieurs centaines de milliers de pixels squelettiques).


### Hypothèses du modèle

**Orientation spatiale :**
- La racine principale est globalement orientée verticalement (axe Y dominant).
- Le système de coordonnées image place Y=0 en haut.
- La racine principale relie l'extrémité la plus haute (Y min) à la plus 
  basse (Y max).

**Continuité temporelle :**
- La racine principale conserve une topologie similaire entre jours consécutifs.
- Les déviations spatiales sont limitées et progressives.
- Un biais de continuité (main_path_bias = 20 par défaut, soit α = 40) force le chemin à suivre la 
  trajectoire du jour précédent.

**Qualité des données :**
- Les masques sont correctement segmentés (racines = pixels blancs).
- La squelettisation préserve la topologie du réseau racinaire.
- Les composantes connexes correspondent à des structures biologiques réelles.

**Robustesse aux cas limites :**
- **Graphes déconnectés** : sélection de la composante avec la plus grande amplitude verticale, avec amplitude verticale (max(Y) − min(Y)).
- **Bifurcations multiples** : BFS garantit la sélection de l'endpoint le plus extrême.
- **Jours manquants** : persistance de la référence temporelle du dernier jour réussi.
- **Déviations latérales** : BFS explore toutes les directions pour atteindre l'extrémité globale.

