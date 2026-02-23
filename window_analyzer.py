"""
VERSION OPTIMISÉE de l'analyseur d'architecture racinaire
Réduction drastique du temps de calcul et de la mémoire utilisée

OPTIMISATIONS PRINCIPALES:
1. Échantillonnage adaptatif pour les grandes images
2. Calcul de distance optimisé (KDTree)
3. Libération agressive de la mémoire
4. Connexion d'objets accélérée
5. Option de résolution réduite
6. Cache des calculs intermédiaires
"""

PYQT_AVAILABLE = False
PYQT_VERSION = None

try:
    from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject, QEvent
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QPushButton, QLabel, QProgressBar, QTextEdit, QGridLayout,
                                QGroupBox, QSpinBox, QFileDialog, QSlider, QSizePolicy,
                                QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
                                QMessageBox, QCheckBox, QComboBox, QScrollArea, QDoubleSpinBox)
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
except:
    try:
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                    QPushButton, QLabel, QProgressBar, QTextEdit, QGridLayout,
                                    QGroupBox, QSpinBox, QFileDialog, QSlider, QSizePolicy,
                                    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
                                    QMessageBox, QCheckBox, QComboBox, QScrollArea, QDoubleSpinBox)
        from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QObject, QEvent
        from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
        PYQT_AVAILABLE = True
        PYQT_VERSION = 5
    except:
        print("This script requires PyQt5 or PyQt6 to run. Neither of these versions was found!")


import cv2, gc, math, os, re, sys, traceback
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tifffile as tiff

from collections import deque
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from widgets import DatasetSelectionWidget

from skimage import morphology, measure, draw, io
from skimage.morphology import skeletonize, thin, convex_hull_image, closing, disk

from scipy.ndimage import convolve, center_of_mass
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from utils import *


def extract_branches_from_graph(G_local):
    """
    Décompose le graphe en 'branches' simples à l'intérieur
    de chaque composante connexe, en couvrant aussi les cycles.
    Retourne une liste de np.array de shape (N_i, 2) (lignes, colonnes).
    """
    branches = []
    visited_edges = set()  # arêtes déjà parcourues
    
    # On travaille composante par composante
    for comp_nodes in nx.connected_components(G_local):
        H = G_local.subgraph(comp_nodes).copy()
        
        # ---------- Étape 1 : partir des extrémités ----------
        for node in H.nodes():
            if H.degree(node) != 1:  # seulement les extrémités
                continue
            
            current = node
            prev = None
            path = [current]
            
            while True:
                neighbors = list(H.neighbors(current))
                # ne prendre que les arêtes non visitées
                next_candidates = [
                    n for n in neighbors
                    if frozenset((current, n)) not in visited_edges
                ]
                
                if not next_candidates:
                    break
                
                nxt = next_candidates[0]
                visited_edges.add(frozenset((current, nxt)))
                
                path.append(nxt)
                prev, current = current, nxt
                
                # arrêt à la prochaine jonction (degré != 2) ou extrémité
                if H.degree(current) != 2:
                    break
            
            if len(path) > 1:
                branches.append(np.array(path))
        
        # ---------- Étape 2 : traiter les cycles / restes ----------
        for (u, v) in H.edges():
            e = frozenset((u, v))
            if e in visited_edges:
                continue
            
            # on part d'une arête restante et on la "déroule"
            current = u
            prev = None
            path = [current]
            
            while True:
                neighbors = list(H.neighbors(current))
                next_candidates = [
                    n for n in neighbors
                    if frozenset((current, n)) not in visited_edges
                ]
                
                if not next_candidates:
                    break
                
                nxt = next_candidates[0]
                visited_edges.add(frozenset((current, nxt)))
                
                path.append(nxt)
                prev, current = current, nxt
                
                # on s'arrête quand il n'y a plus d'arête nouvelle
                # (ici typiquement quand on a bouclé le cycle)
                # pas besoin de condition spéciale : la liste se vide.
                
            if len(path) > 1:
                branches.append(np.array(path))
    
    return branches


def compute_secondary_angles(main_path, secondary_branches):
    """Retourne une liste d'angles (en degrés) pour chaque branche secondaire."""
    abs_angles_deg, angles_deg = [], []
    
    if main_path is None or len(main_path) < 2 or len(secondary_branches) == 0:
        return abs_angles_deg, angles_deg
    
    main_path = np.asarray(main_path)
    
    # 1) angle de la racine principale
    main_vec = main_path[-1] - main_path[0]        # [d_row, d_col]
    main_angle = np.arctan2(main_vec[0], main_vec[1])
    
    for br in secondary_branches:
        br = np.asarray(br)
        if br.shape[0] < 2:
            angles_deg.append(np.nan)
            continue
        
        # 2) base = point le plus proche de la racine principale
        # matrice des distances (N_points_br x N_points_main)
        dists = cdist(br, main_path)
        base_idx, _ = np.unravel_index(np.argmin(dists), dists.shape)
        base = br[base_idx]
        
        # 3) extrémité = point le plus éloigné de la base
        vecs = br - base
        end_idx = np.argmax(np.linalg.norm(vecs, axis=1))
        end = br[end_idx]
        
        branch_vec = end - base
        if np.allclose(branch_vec, 0):
            angles_deg.append(np.nan)
            continue
        
        branch_angle = np.arctan2(branch_vec[0], branch_vec[1])
        
        # 4) angle relatif (en degrés), ramené dans [-180, 180]
        rel = np.degrees(branch_angle - main_angle)
        rel = (rel + 180) % 360 - 180
        
        # souvent on préfère l'angle absolu (0–180°)
        angles_deg.append(rel)
        abs_angles_deg.append(abs(rel))
    
    return abs_angles_deg, angles_deg


def prune_terminal_spurs(skel, min_len_px, protect_mask=None, max_iter=50):
    """
    Remove terminal branches shorter than min_len_px from a skeleton.
    protect_mask: bool array same shape as skel; protected pixels are never removed.
    """
    skel = skel.copy().astype(bool)
    if protect_mask is None:
        protect_mask = np.zeros_like(skel, dtype=bool)
    
    H, W = skel.shape
    
    # 8-neighborhood offsets
    neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    def neighbors(y, x):
        for dy, dx in neigh:
            yy, xx = y+dy, x+dx
            if 0 <= yy < H and 0 <= xx < W and skel[yy, xx]:
                yield (yy, xx)
    
    def degree_map():
        deg = np.zeros_like(skel, dtype=np.uint8)
        ys, xs = np.nonzero(skel)
        for y, x in zip(ys, xs):
            d = 0
            for _ in neighbors(y, x):
                d += 1
            deg[y, x] = d
        return deg
    
    for _ in range(max_iter):
        deg = degree_map()
        endpoints = [(y, x) for (y, x) in zip(*np.nonzero((skel) & (deg == 1)))]
        
        removed_any = False
        
        for ep in endpoints:
            y0, x0 = ep
            if protect_mask[y0, x0]:
                continue
            
            # Walk from endpoint until junction (deg>=3) or until dead end
            path = [(y0, x0)]
            prev = None
            cur = (y0, x0)
            
            while True:
                y, x = cur
                if deg[y, x] >= 3:
                    break  # reached a junction
                
                nxts = [p for p in neighbors(y, x) if p != prev]
                if len(nxts) == 0:
                    break  # isolated / ended
                if len(nxts) > 1:
                    # Shouldn't happen often for deg==2, but safe guard
                    break
                
                prev = cur
                cur = nxts[0]
                path.append(cur)
                
                # If we hit a protected pixel, stop: we don't prune into protected region
                if protect_mask[cur[0], cur[1]]:
                    path = None
                    break
            
            if path is None:
                continue
            
            # If terminal segment is shorter than threshold, remove it
            if len(path) < int(max(1, min_len_px)):
                for (yy, xx) in path:
                    if not protect_mask[yy, xx]:
                        skel[yy, xx] = False
                removed_any = True
        
        if not removed_any:
            break
    
    return skel


class RootArchitectureAnalyzer:
    """Analyseur optimisé pour réduire temps de calcul et mémoire"""
    
    def __init__(self, max_image_size=2000, min_branch_length=10.0, max_skeleton_size=100000, connection_sample_rate=0.05, main_path_bias=20.0, tolerance_pixel=5, grid_rows=3, grid_cols=2, pixels_per_cm=0.0):
        """
        Args:
            max_image_size: Taille max pour redimensionnement (0 = pas de redim)
            min_branch_length: Taille minimale d'une branche pour être comptabilisée en tant que telle
            max_skeleton_size: taille maximale du squelette après laquelle un sous-échantillonnage est imposé
            connection_sample_rate: Taux d'échantillonnage pour connexion (0.01-0.1)
        """
        self.logger_callback = None
        self.max_image_size = max_image_size
        self.min_branch_length = min_branch_length
        self.max_skeleton_size = max_skeleton_size
        self.connection_sample_rate = connection_sample_rate
        self.scale_factor = 1.0
        self.analysis_scale = 1.0
        
        # Attraction vers la racine principale du dernier jour
        self.main_path_bias = main_path_bias # force de la pénalisation
        
        # Rayon de fermeture appliqué au masque en cours, résultat sur lequel on applique l'intersection avec le masque précédent pour la fusion
        self.tolerance_pixel = tolerance_pixel
        
        # Nombre de lignes et colonnes grille
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Conversion pixels -> cm
        self.pixels_per_cm = pixels_per_cm
        self.convert_to_cm = self.pixels_per_cm > 0.0
        
    def log(self, message):
        if self.logger_callback:
            self.logger_callback(message)
        else:
            print(message)
    
    def resize_if_needed(self, image):
        """Redimensionne l'image si trop grande"""
        if self.max_image_size <= 0:
            self.analysis_scale = 1.0
            self.scale_factor = 1.0
            return image
        
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > self.max_image_size:
            self.analysis_scale = self.max_image_size / max_dim
            self.scale_factor = self.analysis_scale
            new_h = int(h * self.analysis_scale)
            new_w = int(w * self.analysis_scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            self.log(f"Image resizing: {w}x{h} -> {new_w}x{new_h} (factor: {self.analysis_scale:.2f})")
            return resized
        
        self.analysis_scale = 1.0
        self.scale_factor = 1.0
        return image
    
    def preprocess_mask(self, binary_mask, closing_radius=20, min_size=500):
        """Prétraitement optimisé"""
        # Fermeture morphologique
        processed = closing(binary_mask.astype(bool), disk(closing_radius))
        
        # Suppression des petits objets
        processed = morphology.remove_small_objects(processed, min_size=min_size)
        
        return processed.astype(np.uint8)
    
    def merge_and_connect_objects(self, binary_mask, line_thickness=5, max_iterations=10, max_connection_distance=None):
        """
        Version OPTIMISÉE de la connexion d'objets
        Utilise KDTree pour accélérer la recherche de voisins
        """
        current_mask = binary_mask.copy()
        iteration = 0
        
        # distance max par défaut (en pixels)
        if max_connection_distance is None:
            # quelque chose de raisonnable pour recoller de petites ruptures
            max_connection_distance = round(0.05 * min(binary_mask.shape[0], binary_mask.shape[1]))
            print(f"max_connection_distance = {max_connection_distance}")
        
        while iteration < max_iterations:
            # Étiquetage
            label_image = measure.label(current_mask)
            regions = measure.regionprops(label_image)
            
            if len(regions) <= 1:
                self.log(f"Merging complete: {iteration} iterations")
                break
            
            self.log(f"Iteration {iteration}: {len(regions)} objects")
            
            # Extraire les contours (plus rapide que tous les coords)
            region_contours = []
            for region in regions:
                # Prendre seulement les points du contour, pas tous les pixels
                coords = region.coords
                # Échantillonnage adaptatif
                sample_size = max(10, int(len(coords) * self.connection_sample_rate))
                indices = np.random.choice(len(coords), size=min(sample_size, len(coords)), replace=False)
                region_contours.append(coords[indices].astype(np.float32))
            
            # Pour chaque région, trouver la plus proche avec KDTree
            connections = []
            for i, coords_i in enumerate(region_contours):
                if len(coords_i) == 0:
                    continue
                
                min_dist = np.inf
                best_connection = None
                
                for j, coords_j in enumerate(region_contours):
                    if i >= j or len(coords_j) == 0:
                        continue
                    
                    # KDTree pour recherche rapide du plus proche voisin
                    tree = cKDTree(coords_j)
                    distances, indices = tree.query(coords_i, k=1)
                    min_idx = np.argmin(distances)
                    
                    if distances[min_idx] < min_dist:
                        min_dist = distances[min_idx]
                        best_connection = (coords_i[min_idx], coords_j[indices[min_idx]])
                
                if (best_connection is not None) and (min_dist <= max_connection_distance):
                    connections.append(best_connection)
            
            # Dessiner toutes les connexions
            for point1, point2 in connections:
                rr, cc = draw.line(
                    int(point1[0]), int(point1[1]),
                    int(point2[0]), int(point2[1])
                )
                
                # Épaissir la ligne
                for r, c in zip(rr, cc):
                    rr_thick, cc_thick = draw.disk(
                        (r, c), radius=line_thickness, 
                        shape=current_mask.shape
                    )
                    current_mask[rr_thick, cc_thick] = True
            
            iteration += 1
            
            # Libération mémoire
            del label_image, regions, region_contours, connections
            gc.collect()
        
        return current_mask
    
    
    def _compute_exact_skeleton_length(self, skeleton_bool):
        """
        Calcule la longueur exacte du squelette en pixels en parcourant tous les segments.
        Utilise la 8-connectivité avec distances euclidiennes exactes.
        Version OPTIMISÉE avec code vectorisé numpy.
        
        Returns:
            float: Longueur totale du squelette en pixels
        """
        if skeleton_bool is None or skeleton_bool.size == 0:
            return 0.0
        
        h, w = skeleton_bool.shape[:2]
        if h == 0 or w == 0:
            return 0.0
        
        # Version vectorisée : utiliser des opérations numpy au lieu de boucles Python
        total_length = 0.0
        
        # Pour éviter de compter deux fois les mêmes arêtes, on ne regarde que
        # les voisins vers la droite et le bas (4 directions)
        
        # Direction 1 : droite (0, 1) - distance 1.0
        if w > 1:
            right_edges = skeleton_bool[:, :-1] & skeleton_bool[:, 1:]
            total_length += np.sum(right_edges) * 1.0
        
        # Direction 2 : bas (1, 0) - distance 1.0  
        if h > 1:
            down_edges = skeleton_bool[:-1, :] & skeleton_bool[1:, :]
            total_length += np.sum(down_edges) * 1.0
        
        # Direction 3 : diagonale bas-droite (1, 1) - distance sqrt(2)
        if h > 1 and w > 1:
            diag_br_edges = skeleton_bool[:-1, :-1] & skeleton_bool[1:, 1:]
            total_length += np.sum(diag_br_edges) * math.sqrt(2.0)
        
        # Direction 4 : diagonale bas-gauche (1, -1) - distance sqrt(2)
        if h > 1 and w > 1:
            diag_bl_edges = skeleton_bool[:-1, 1:] & skeleton_bool[1:, :-1]
            total_length += np.sum(diag_bl_edges) * math.sqrt(2.0)
        
        return total_length
    
    
    def _grid_edge_lengths_from_skeleton(self, skeleton_bool, grid_rows, grid_cols):
        """Compute skeleton length per grid cell (grid_rows x grid_cols), fast & vectorized.
        
        We count each 8-connected edge once using forward neighbors:
        right, down, down-right, down-left. Each edge is assigned to a cell
        by the midpoint of its segment.
        
        Returns a (grid_rows, grid_cols) float array in pixel units.
        """
        if grid_rows is None or grid_cols is None:
            return None
        try:
            grid_rows = int(grid_rows)
            grid_cols = int(grid_cols)
        except Exception:
            return None
        if grid_rows <= 0 or grid_cols <= 0:
            return None
        
        h, w = skeleton_bool.shape[:2]
        if h == 0 or w == 0:
            return None
        
        sk = skeleton_bool.astype(bool, copy=False)
        out = np.zeros((grid_rows, grid_cols), dtype=np.float64)
        
        # cell size in pixels (float)
        cell_h = h / float(grid_rows)
        cell_w = w / float(grid_cols)
        if cell_h <= 0 or cell_w <= 0:
            return out
        
        sqrt2 = math.sqrt(2.0)
        
        def _accumulate(edge_mask, dy, dx, seg_len):
            if edge_mask is None:
                return
            ys, xs = np.nonzero(edge_mask)
            if ys.size == 0:
                return
            
            # midpoint of the segment
            my = ys.astype(np.float64) + 0.5 * float(dy)
            mx = xs.astype(np.float64) + 0.5 * float(dx)
            
            rr = np.floor(my / cell_h).astype(np.int32)
            cc = np.floor(mx / cell_w).astype(np.int32)
            
            # clamp
            rr = np.clip(rr, 0, grid_rows - 1)
            cc = np.clip(cc, 0, grid_cols - 1)
            
            np.add.at(out, (rr, cc), seg_len)
        
        # Count each edge once: right, down, diag down-right, diag down-left
        if w > 1:
            _accumulate(sk[:, :-1] & sk[:, 1:], 0, 1, 1.0)
        if h > 1:
            _accumulate(sk[:-1, :] & sk[1:, :], 1, 0, 1.0)
        if h > 1 and w > 1:
            _accumulate(sk[:-1, :-1] & sk[1:, 1:], 1, 1, math.sqrt(2.0))
            _accumulate(sk[:-1, 1:] & sk[1:, :-1], 1, -1, math.sqrt(2.0))
        
        return out
    
    
    def skeletonize_and_analyze(self, binary_mask, main_ref_points=None, main_ref_path=None, grid_rows=None, grid_cols=None):
        """Version optimisée de l'analyse du squelette
        
        Args:
        
            binary_mask : np.ndarray, Masque binaire de la racine.
            main_ref_points : (p0, p1) ou None, Deux points (haut, bas) de la racine principale globale.
            main_ref_path : np.ndarray (N, 2) ou None, Chemin complet de la racine principale du dernier jour ;
                            utilisé pour définir un couloir spatial autour duquel on contraint la racine principale.
        """
        
        
        # Squelettisation
        # IMPORTANT:
        # - skimage.skeletonize() (Zhang-Suen) a un comportement surprenant sur certains objets
        #   déjà très fins (ex: bande diagonale de 2px), pouvant "s'effondrer" en un seul pixel.
        # - Dans ces cas, thin() (amincissement) est beaucoup plus stable.
        # On fait donc un fallback automatique si le squelette est anormalement petit.
        mask_bool = binary_mask.astype(bool)
        skeleton = skeletonize(mask_bool)
        
        try:
            # Heuristique: si le squelette est trop petit par rapport au masque, on fallback
            msum = int(mask_bool.sum())
            ssum = int(skeleton.sum())
            if msum > 0 and ssum < max(2, int(0.01 * msum)):
                skeleton = thin(mask_bool)
        except Exception:
            # En cas de souci d'import/compatibilité, on garde skeletonize()
            pass
        skeleton_points = np.array(np.nonzero(skeleton)).T
        
        if len(skeleton_points) == 0:
            return self._empty_features()
        
        
        # Détection des extrémités (optimisée)
        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant') - skeleton.astype(np.uint8)
        endpoints = (skeleton & (neighbor_count == 1))
        endpoint_coords = np.array(np.nonzero(endpoints)).T
        
        endpoint_count_raw = int(len(endpoint_coords))
        root_count_raw = max(0, endpoint_count_raw - 1)
        
        # Si trop de points, échantillonner le graphe
        if len(skeleton_points) > self.max_skeleton_size:
            self.log(f"Skeleton thick ({len(skeleton_points)} points), sampling...")
            # Garder tous les endpoints + échantillonnage du reste
            non_endpoint_mask = skeleton & (~endpoints)
            non_endpoint_points = np.array(np.nonzero(non_endpoint_mask)).T
            
            if len(non_endpoint_points) > 50000:
                sample_indices = np.random.choice(
                    len(non_endpoint_points), 
                    size=5000, 
                    replace=False
                )
                sampled_points = non_endpoint_points[sample_indices]
                skeleton_points = np.vstack([endpoint_coords, sampled_points])
            
            self.log(f"Points reduced to {len(skeleton_points)}")
        
        # Création du graphe (rapide, 8-connectivité en O(N))
        # L'ancienne version utilisait un KDTree + query_ball_point par nœud, ce qui peut devenir très lent
        # sur certains masques (ex: probabilités / squelette dense). Ici on construit les arêtes directement
        # via les 4 voisins "avant" (droite, bas, diag bas-droite, diag bas-gauche) pour éviter les doublons.
        G = nx.Graph()
        
        # Ensemble de points pour membership O(1)
        pts = [tuple(p) for p in skeleton_points]
        pts_set = set(pts)
        
        # Ajouter les nœuds
        G.add_nodes_from(pts)
        
        # Ajouter les arêtes (8-connectivité, sans doublons)
        sqrt2 = math.sqrt(2.0)
        neighbor_steps = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, sqrt2), (1, -1, sqrt2)]
        for (y, x) in pts:
            for dy, dx, w in neighbor_steps:
                nb = (y + dy, x + dx)
                if nb in pts_set:
                    G.add_edge((y, x), nb, weight=w)
        
        # ---------- Biais vers la racine de référence (dernier jour) ----------
        node_dist = None
        if main_ref_path is not None and len(G.nodes) >= 2:
            ref = np.asarray(main_ref_path, dtype=float)
            if ref.ndim == 2 and ref.shape[0] >= 2:
                node_dist = {}
                for n in G.nodes:
                    p = np.array(n, dtype=float)
                    d = self._distance_point_to_path(p, ref)
                    node_dist[n] = d
                
                # échelle de distance pour normaliser (évite des énormes nombres)
                max_d = max(node_dist.values()) or 1.0
                alpha = float(self.main_path_bias) * 2.0  # Doublé pour stabilité
                
                # poids "biaisé" pour chaque arête
                for u, v, data in G.edges(data=True):
                    base_len = data.get(
                        "weight",
                        np.linalg.norm(np.array(u, dtype=float) - np.array(v, dtype=float))
                    )
                    d = (node_dist[u] + node_dist[v]) / 2.0
                    penalty = 1.0 + alpha * (d / max_d) ** 2
                    data["biased_weight"] = base_len * penalty
            else:
                node_dist = None
        
        # Calcul du chemin principal
        main_path_length = 0.0
        main_path = []
        
        # 1) Essai avec la racine de référence (dernier jour)
        if (main_ref_points is not None or main_ref_path is not None) and len(G.nodes) >= 2:
            try:
                if main_ref_points is not None:
                    ref_top, ref_bottom = main_ref_points
                else:
                    ref_top, ref_bottom = main_ref_path[0], main_ref_path[-1]
                
                ref_top = np.asarray(ref_top, dtype=float)
                ref_bottom = np.asarray(ref_bottom, dtype=float)
                
                start_node = min(G.nodes, key=lambda n: np.linalg.norm(np.array(n, dtype=float) - ref_top))
                end_node = min(G.nodes, key=lambda n: np.linalg.norm(np.array(n, dtype=float) - ref_bottom))
                
                if nx.has_path(G, start_node, end_node):
                    # si on a un biais (main_ref_path), on l'utilise
                    weight_attr = "biased_weight" if main_ref_path is not None else "weight"
                    main_path = nx.shortest_path(G,
                                                 source=start_node,
                                                 target=end_node,
                                                 weight=weight_attr)
                    
                    # prolonger jusqu’aux vraies extrémités
                    main_path = self._extend_path_to_endpoints(G, main_path)
                    
                    # prolonger le chemin le long de l'axe global (ref_top -> ref_bottom)
                    #                     main_path = self._extend_path_along_axis(G, main_path, ref_top, ref_bottom)
                    
                    main_path_length = sum(
                        np.linalg.norm(np.array(main_path[i], dtype=float) -
                                       np.array(main_path[i-1], dtype=float))
                        for i in range(1, len(main_path))
                    )
                    G.remove_nodes_from(main_path)
            except Exception:
                main_path = []
                main_path_length = 0.0
        
        # 2) Si ca n'a pas marche : chercher dans chaque composante connexe separement.
        #    Strategie : pour chaque composante, tracer chemin endpoint_haut -> endpoint_bas,
        #    garder la composante avec la plus grande amplitude verticale.
        #    Gere le cas ou la segmentation cree un graphe deconnecte.
        if main_path_length == 0.0 and len(G.nodes) >= 2:
            try:
                best_path = []
                best_length = 0.0
                
                components = sorted(
                    nx.connected_components(G),
                    key=lambda comp: max(n[0] for n in comp) - min(n[0] for n in comp),
                    reverse=True
                )
                
                for comp_nodes in components:
                    if len(comp_nodes) < 2:
                        continue
                    G_comp = G.subgraph(comp_nodes)
                    nodes_arr = np.array(list(comp_nodes), dtype=float)
                    
                    ep_comp = [n for n in comp_nodes if G_comp.degree(n) == 1]
                    if len(ep_comp) >= 2:
                        ep_arr = np.array(ep_comp, dtype=float)
                        s = ep_comp[int(np.argmin(ep_arr[:, 0]))]
                        e = ep_comp[int(np.argmax(ep_arr[:, 0]))]
                    else:
                        s = tuple(nodes_arr[int(np.argmin(nodes_arr[:, 0]))].astype(int))
                        e = tuple(nodes_arr[int(np.argmax(nodes_arr[:, 0]))].astype(int))
                    
                    if s == e or not nx.has_path(G_comp, s, e):
                        continue
                    
                    candidate = nx.shortest_path(G_comp, source=s, target=e, weight="weight")
                    candidate = self._extend_path_to_endpoints(G_comp, candidate)
                    candidate_length = sum(
                        np.linalg.norm(np.array(candidate[idx]) - np.array(candidate[idx-1]))
                        for idx in range(1, len(candidate))
                    )
                    
                    if candidate_length > best_length:
                        best_length = candidate_length
                        best_path = candidate
                
                if best_path:
                    main_path = best_path
                    main_path_length = best_length
                    G.remove_nodes_from(main_path)
            
            except Exception:
                pass
        
        
        
        # ============================
        # Comptage des terminaisons après élagage des petites terminaisons
        # (protection de main_path pour ne pas élagaguer la racine principale)
        # ============================
        
        endpoint_count_pruned = endpoint_count_raw
        root_count_pruned = root_count_raw
        endpoint_coords_pruned = endpoint_coords  # fallback
        
        if isinstance(skeleton, np.ndarray) and skeleton.size > 0 and (main_path is not None) and (len(main_path) >= 2):
            protect_mask = np.zeros_like(skeleton, dtype=bool)
            for (yy, xx) in main_path:
                yy = int(yy); xx = int(xx)
                if 0 <= yy < protect_mask.shape[0] and 0 <= xx < protect_mask.shape[1]:
                    protect_mask[yy, xx] = True
            
            # seuil en pixels de la résolution actuelle (comme pour tes branches)
            prune_len_px = max(1, int(round(self.min_branch_length * self.scale_factor)))
            
            skeleton_pruned = prune_terminal_spurs(
                skeleton,
                min_len_px=prune_len_px,
                protect_mask=protect_mask,
                max_iter=50
            )
            
            # recompute endpoints on pruned skeleton
            neighbor_count_p = convolve(skeleton_pruned.astype(np.uint8), kernel, mode='constant') - skeleton_pruned.astype(np.uint8)
            endpoints_p = (skeleton_pruned & (neighbor_count_p == 1))
            endpoint_coords_pruned = np.array(np.nonzero(endpoints_p)).T
            
            endpoint_count_pruned = int(len(endpoint_coords_pruned))
            root_count_pruned = max(0, endpoint_count_pruned - 1)
        else:
            skeleton_pruned = skeleton  # pour éviter NameError si tu veux l'afficher/debug
        
        
        # Extraire les branches sur le graphe restant (sans la racine principale)
        secondary_branches = extract_branches_from_graph(G)
        
        # --- FILTRAGE DES PETITES BRANCHES ---
        def _branch_length_px(br):
            br = np.asarray(br, dtype=float)
            if br.shape[0] < 2:
                return 0.0
            
            diffs = np.diff(br, axis=0)
            seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
            return float(seg_lengths.sum())
        
        if self.min_branch_length > 0:
            # self.min_branch_length est exprimé en "pixels de l'image d'origine".
            # Comme on a éventuellement redimensionné l'image avec self.scale_factor,
            # on corrige le seuil pour la résolution actuelle :
            length_threshold_resized = self.min_branch_length * self.scale_factor
            filtered_branches = []
            for br in secondary_branches:
                if _branch_length_px(br) >= length_threshold_resized:
                    filtered_branches.append(br)
            secondary_branches = filtered_branches
        # --- FIN FILTRAGE ---
        
        # Calcul des angles des racines secondaires
        abs_secondary_angles, secondary_angles = compute_secondary_angles(main_path, secondary_branches)
        mean_angles, std_angles = np.nan, np.nan
        mean_abs_angles, std_abs_angles = np.nan, np.nan
        if len(secondary_angles) > 0:
            mean_angles = np.mean(secondary_angles)
            std_angles = np.std(secondary_angles)
        if len(abs_secondary_angles) > 0:
            mean_abs_angles = np.mean(abs_secondary_angles)
            std_abs_angles = np.std(abs_secondary_angles)
        
        # Longueur totale des branches secondaires
        secondary_length = 0.0
        for branch in secondary_branches:
            if len(branch) < 2:
                continue
            secondary_length += sum(
                np.linalg.norm(np.array(branch[i]) - np.array(branch[i - 1]))
                for i in range(1, len(branch))
            )
        
        branch_count = len(secondary_branches)
        
        # Root count attach
        attach_positions = []
        
        if main_path is not None and len(main_path) > 1:
            main_arr = np.asarray(main_path, dtype=float)
            
            # longueur cumulée le long du tronc
            diffs = np.diff(main_arr, axis=0)
            seg_len = np.linalg.norm(diffs, axis=1)
            cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
            total_len = cumlen[-1] if len(cumlen) > 0 else 1.0
            
            max_attach_dist = 10.0  # distance max en pixels entre branche et tronc
            
            for br in secondary_branches:  # ou secondary_level1
                br_arr = np.asarray(br, dtype=float)
                # distance de chaque point de la branche au tronc
                # (si trop lourd, tu peux échantillonner quelques points)
                dists = np.linalg.norm(
                    br_arr[:, None, :] - main_arr[None, :, :],
                    axis=2
                )
                flat_idx = np.argmin(dists)  # index aplati sur (n_branch_pts * n_main_pts)
                i_br, i_main = np.unravel_index(flat_idx, dists.shape)  # (idx point branche, idx point tronc)
                min_dist = dists[i_br, i_main]
                
                if min_dist > max_attach_dist:
                    continue
                
                # position le long du tronc normalisée [0,1] -> on indexe cumlen avec i_main (pas flat_idx)
                attach_pos = cumlen[i_main] / max(total_len, 1e-6)
                attach_positions.append(attach_pos)
        
        # regrouper les positions proches
        attach_positions = np.array(sorted(attach_positions))
        cluster_eps = 0.02  # 2% de la longueur du tronc, par ex.
        
        root_count_attach = 0
        if len(attach_positions) > 0:
            root_count_attach = 1  # la racine principale
            current_start = attach_positions[0]
            for p in attach_positions[1:]:
                if p - current_start > cluster_eps:
                    root_count_attach += 1
                    current_start = p
        
        # Autres caractéristiques
        convex_hull = convex_hull_image(binary_mask)
        convex_area = np.sum(convex_hull)
        
        if np.sum(skeleton) > 0:
            centroid_y, centroid_x = center_of_mass(skeleton)
        else:
            centroid_x, centroid_y = 0, 0
        
        # Ajuster les mesures selon le facteur d'échelle
        scale = 1.0 / self.scale_factor if self.scale_factor > 0 else 1.0
        
        # Longueur par cellule de grille
        grid_lengths_cm = None
        grid_lengths = self._grid_edge_lengths_from_skeleton(skeleton.astype(bool), self.grid_rows, self.grid_cols)
        if grid_lengths is not None:
            grid_lengths = grid_lengths * scale  # same scaling as lengths
            if self.convert_to_cm and self.pixels_per_cm and self.pixels_per_cm > 0:
                grid_lengths_cm = grid_lengths / float(self.pixels_per_cm)
        
        exact_skeleton_length = self._compute_exact_skeleton_length(skeleton_pruned.astype(bool)) * scale
        
        result = {
            'main_root_length': main_path_length * scale,
            'secondary_roots_length': secondary_length * scale,
            'total_root_length': (main_path_length + secondary_length) * scale,
            'exact_skeleton_length': exact_skeleton_length,
            'branch_count': branch_count,
            'root_count_attach': root_count_attach,
            'endpoint_count_raw': endpoint_count_raw,
            'root_count_raw': root_count_raw,
            'endpoint_count': endpoint_count_pruned,
            'root_count': root_count_pruned,
            'convex_area': convex_area * (scale ** 2),
            'total_area': np.sum(binary_mask) * (scale ** 2),
            'centroid_x': centroid_x * scale,
            'centroid_y': centroid_y * scale,
            'centroid_x_display': centroid_x,
            'centroid_y_display': centroid_y,
            'skeleton': skeleton,
            'main_path': np.array(main_path) if main_path else np.array([]),
            'endpoints': endpoint_coords,
            'convex_hull': convex_hull,
            'secondary_branches': np.array(secondary_branches, dtype=object),
            'secondary_angles': np.array(secondary_angles, dtype=float),
            'mean_secondary_angles': mean_angles,
            'std_secondary_angles': std_angles,
            'abs_secondary_angles': np.array([abs_secondary_angles], dtype=float),
            'mean_abs_secondary_angles': mean_abs_angles,
            'std_abs_secondary_angles': std_abs_angles,
            'scale':scale,
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'grid_lengths': grid_lengths,
            'grid_lengths_cm': grid_lengths_cm,
            
            # Debug
            'endpoints_raw': endpoint_coords,
            'endpoints_pruned': endpoint_coords_pruned,
            'skeleton_pruned': skeleton_pruned
        }
        
        if self.convert_to_cm:
            result['pixels_per_cm'] = self.pixels_per_cm
            result['main_root_length_cm'] = result['main_root_length'] / self.pixels_per_cm
            result['secondary_roots_length_cm'] = result['secondary_roots_length'] / self.pixels_per_cm
            result['total_root_length_cm'] = result['total_root_length'] / self.pixels_per_cm
            result['exact_skeleton_length_cm'] = result['exact_skeleton_length'] / self.pixels_per_cm
            result['total_area'] = result['exact_skeleton_length'] / (self.pixels_per_cm ** 2)
            result['convex_hull_cm'] = result['convex_hull'] / (self.pixels_per_cm ** 2)
            result['convex_area_cm'] = result['convex_area'] / (self.pixels_per_cm ** 2)
        
        # Libération mémoire
        del skeleton_points, G
        gc.collect()
        
        return result
    
    def _extend_path_to_endpoints(self, G, path):
        """
        Prolonge un chemin existant jusqu'aux vraies extrémités globales.
        
        Stratégie : depuis chaque extrémité du chemin, effectue un parcours
        en avant uniquement (BFS unidirectionnel sans revenir en arrière)
        pour atteindre l'endpoint le plus extrême, en évitant de croiser
        des nœuds déjà dans le chemin principal.
        """
        if not path or len(path) < 2:
            return path
        
        def find_extreme_endpoint_forward(G, anchor, forbidden, extreme='max_y'):
            """
            BFS depuis anchor, sans jamais passer par les nœuds forbidden.
            Retourne le chemin (liste) vers l'endpoint le plus extrême trouvé.
            """
            # BFS pour explorer tous les nœuds accessibles sans revenir
            visited = {anchor}
            # parent dict pour reconstruire le chemin
            parent = {anchor: None}
            queue = deque([anchor])
            
            while queue:
                curr = queue.popleft()
                for nb in G.neighbors(curr):
                    if nb not in visited and nb not in forbidden:
                        visited.add(nb)
                        parent[nb] = curr
                        queue.append(nb)
            
            # Parmi tous les nœuds visités, trouver les endpoints (degré 1 dans G)
            endpoints_reached = [n for n in visited if G.degree(n) == 1 and n != anchor]
            
            if not endpoints_reached:
                # Pas d'endpoint : prendre le nœud le plus extrême parmi les visités
                candidates = list(visited - {anchor})
                if not candidates:
                    return []
                endpoints_reached = candidates
            
            # Choisir l'endpoint le plus extrême
            if extreme == 'min_y':
                target = min(endpoints_reached, key=lambda n: n[0])
            else:
                target = max(endpoints_reached, key=lambda n: n[0])
            
            # Reconstruire le chemin depuis anchor vers target via parent
            path_ext = []
            curr = target
            while curr is not None:
                path_ext.append(curr)
                curr = parent[curr]
            path_ext.reverse()  # anchor → target
            
            return path_ext  # commence par anchor
        
        path_list = list(path)
        path_set = set(tuple(n) for n in path_list)
        
        # --- Prolonger vers le HAUT (endpoint avec Y minimum) ---
        top_anchor = tuple(path_list[0])
        # Forbidden = tout le chemin sauf l'ancre elle-même
        forbidden_top = path_set - {top_anchor}
        ext_top = find_extreme_endpoint_forward(G, top_anchor, forbidden_top, extreme='min_y')
        if len(ext_top) > 1:
            # ext_top[0] == top_anchor, on l'exclut pour éviter le doublon
            path_list = list(reversed(ext_top[1:])) + path_list
        
        # Recalculer path_set après extension haute
        path_set = set(tuple(n) for n in path_list)
        
        # --- Prolonger vers le BAS (endpoint avec Y maximum) ---
        bottom_anchor = tuple(path_list[-1])
        forbidden_bot = path_set - {bottom_anchor}
        ext_bot = find_extreme_endpoint_forward(G, bottom_anchor, forbidden_bot, extreme='max_y')
        if len(ext_bot) > 1:
            path_list = path_list + list(ext_bot[1:])
        
        return path_list
    
    def _extend_path_along_axis(self, G, path, top, bottom):
        """
        Étend un chemin 'path' le long de l'axe global (top -> bottom)
        jusqu'aux extrémités atteignables dans le graphe.
        
        On utilise l’axe global (top -> bottom) plutôt que la seule
        direction locale pour rester collé au tronc principal même
        dans une zone très ramifiée.
        """
        if not path or len(path) < 2:
            return path
        
        path_nodes = list(path)
        
        top = np.asarray(top, dtype=float)
        bottom = np.asarray(bottom, dtype=float)
        axis = bottom - top
        if np.allclose(axis, 0):
            return path_nodes
        axis = axis / np.linalg.norm(axis)
        
        def extend_from_end(idx_curr, idx_prev, direction_sign):
            nonlocal path_nodes
            prev = np.array(path_nodes[idx_prev], dtype=float)
            curr = np.array(path_nodes[idx_curr], dtype=float)
            dir_ref = direction_sign * axis
            
            while True:
                curr_t = tuple(curr.astype(int))
                prev_t = tuple(prev.astype(int))
                
                # voisins sauf le nœud précédent
                neighbors = [n for n in G.neighbors(curr_t) if n != prev_t]
                if not neighbors:
                    break  # extrémité atteinte
                
                best_n = None
                best_dp = -1e9
                for n in neighbors:
                    v = np.array(n, dtype=float) - curr
                    if np.allclose(v, 0):
                        continue
                    v = v / np.linalg.norm(v)
                    dp = float(np.dot(v, dir_ref))  # alignement avec l'axe global
                    if dp > best_dp:
                        best_dp = dp
                        best_n = n
                
                if best_n is None:
                    break
                
                if direction_sign < 0:
                    # on prolonge vers le haut : on ajoute au début
                    path_nodes.insert(0, best_n)
                    idx_curr, idx_prev = 0, 1
                else:
                    # on prolonge vers le bas : on ajoute à la fin
                    path_nodes.append(best_n)
                    idx_curr, idx_prev = len(path_nodes) - 1, len(path_nodes) - 2
                
                prev = curr
                curr = np.array(best_n, dtype=float)
        
        # prolonger vers le "haut" du chemin (direction -axis)
        extend_from_end(0, 1, -1.0)
        # prolonger vers le "bas" du chemin (direction +axis)
        extend_from_end(len(path_nodes) - 1, len(path_nodes) - 2, +1.0)
        
        return path_nodes
    
    def _distance_point_to_path(self, point, path_array):
        """Distance euclidienne min entre un point (2,) et un chemin (N,2)."""
        diffs = path_array - point
        return float(np.min(np.linalg.norm(diffs, axis=1)))
    
    def _empty_features(self):
        """Features vides"""
        return {
            'main_root_length': 0.0,
            'secondary_roots_length': 0.0,
            'total_root_length': 0.0,
            'exact_skeleton_length': 0.0,
            'branch_count': 0,
            'root_count_attach': 0,
            'endpoint_count_raw': 0,
            'root_count_raw': 0,
            'endpoint_count': 0,
            'root_count': 0,
            'convex_area': 0,
            'total_area': 0,
            'centroid_x': 0.0,
            'centroid_y': 0.0,
            'centroid_x_display': 0.0,
            'centroid_y_display': 0.0,
            'skeleton': np.array([]),
            'main_path': np.array([]),
            'endpoints': np.array([]),
            'convex_hull': np.array([]),
            'secondary_branches': np.array([], dtype=object),
            'secondary_angles': np.array([], dtype=float),
            'mean_secondary_angles': 0.0,
            'std_secondary_angles': 0.0,
            'abs_secondary_angles': np.array([], dtype=float),
            'mean_abs_secondary_angles': 0.0,
            'std_abs_secondary_angles': 0.0,
            'scale':1.0,
            'grid_rows': 0,
            'grid_cols': 0,
            # Debug
            'endpoints_raw': np.array([]),
            'endpoints_pruned': np.array([]),
            'skeleton_pruned': np.array([])
        }


class RootArchitectureWorker(QThread):
    """Worker optimisé pour l'analyse"""
    
    progress = pyqtSignal(int, str)
    day_analyzed = pyqtSignal(int, dict, np.ndarray, np.ndarray)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, mask_files, params):
        super().__init__()
        self.mask_files = sorted(mask_files)
        self.params = params
        
        # Créer l'analyseur avec les paramètres d'optimisation
        self.analyzer = RootArchitectureAnalyzer(
            max_image_size=params.get('max_image_size', 2000),
            min_branch_length=params.get('min_branch_length', 20.0),
            max_skeleton_size=params.get('min_sampling_threshold', 100000),
            connection_sample_rate=params.get('connection_sample_rate', 0.05),
            main_path_bias=params.get('main_path_bias', 20.0),
            tolerance_pixel=params.get('fusion_tolerance_pixel', 5),
            grid_rows=params.get('grid_rows', 3),
            grid_cols=params.get('grid_cols', 2),
            pixels_per_cm=params.get('pixels_per_cm', 0.0)
        )
        self.analyzer.logger_callback = self.log_message
        
    def log_message(self, message):
        print(message)
    
    def run(self):
        """Exécution optimisée de l'analyse (avec racine principale globale)"""
        try:
            def _check_interrupt():
                try:
                    if self.isInterruptionRequested():
                        raise InterruptedError("Stopped by user")
                except AttributeError:
                    return
            
            # ----------------------------
            # 1) Pré-traitement de tous les jours
            # ----------------------------
            preprocessed = []
            previous_merged = None
            ref_path = None  # CORRECTION : Initialiser ref_path avant son utilisation
            total = len(self.mask_files)
            
            for idx, mask_file in enumerate(self.mask_files):
                _check_interrupt()
                filename = Path(mask_file).stem
                # 0 -> 50% pendant le pré-traitement. Le +1 permet d'atteindre 50% sur le dernier item.
                preprocess_progress = int(((idx + 1) / max(total, 1)) * 50)
                preprocess_progress = max(0, min(50, preprocess_progress))
                self.progress.emit(preprocess_progress, f"Preprocessing: {filename}")
                
                # Charger l'image
                if mask_file.endswith((os.extsep + 'tif', os.extsep + 'tiff')):
                    mask = tiff.imread(mask_file)
                    if mask.ndim > 2:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    # Safety checks
                    if mask.ndim != 2:
                        raise ValueError(f"Mask must be 2D after conversion, got ndim={mask.ndim} for {mask_file}")
                    
                else:
                    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError(f"Could not read image: {mask_file}")
                    if mask.ndim != 2:
                        raise ValueError(f"Mask must be 2D after reading, got ndim={mask.ndim} for {mask_file}")
                
                # Redimensionner si nécessaire
                mask = self.analyzer.resize_if_needed(mask)
                
                # Binarisation
                # NOTE: certains fichiers "*_Probabilities*" ne sont pas des masques binaires stricts.
                # Un simple (mask > 0) peut alors transformer tout le fond en "racine" et faire exploser
                # la taille du squelette (et donc le temps de calcul).
                m = mask
                try:
                    # Si l'image est déjà binaire (0/1 ou 0/255), on garde le comportement simple.
                    uniq = np.unique(m)
                    if uniq.size <= 2:
                        binary_mask = (m > 0).astype(np.uint8)
                    else:
                        # Otsu sur uint8 (robuste pour des cartes de probabilités / gradients)
                        if m.dtype != np.uint8:
                            m8 = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        else:
                            m8 = m
                        _, th = cv2.threshold(m8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binary_mask = th.astype(np.uint8)
                        
                        # Fallback si Otsu produit un masque quasi plein (souvent dû à un fond non nul)
                        fill_ratio = float(np.mean(binary_mask))
                        if fill_ratio > 0.95:
                            # seuil plus strict
                            _, th2 = cv2.threshold(m8, 128, 1, cv2.THRESH_BINARY)
                            binary_mask = th2.astype(np.uint8)
                            fill_ratio2 = float(np.mean(binary_mask))
                            if fill_ratio2 > 0.95:
                                # dernier recours : percentile (garde les valeurs les plus fortes)
                                t = np.percentile(m8, 98)
                                binary_mask = (m8 >= t).astype(np.uint8)
                except Exception:
                    # Comportement historique
                    binary_mask = (m > 0).astype(np.uint8)
                
                original_mask = binary_mask.copy()
                
                # Métadonnées
                day_match = re.search(r"_J(\d+)_", filename)
                day = int(day_match.group(1)) if day_match else idx
                modality = filename.split("_")[0] if "_" in filename else "unknown"
                
                # Fusion temporelle (VERSION CORRIGÉE - continuité locale)
                # Ne plus accumuler infiniment les pixels, mais permettre une continuité locale
                # NOTE : Si graphe déconnecté (trous), augmenter tolerance_pixels (ex: 8-10)
                #        ou activer "Connect objects" avec max_connection_dst > 150
                if previous_merged is not None and self.params['temporal_merge']:
                    # Redimensionner previous_merged si nécessaire
                    if previous_merged.shape != binary_mask.shape:
                        previous_merged = cv2.resize(
                            previous_merged,
                            (binary_mask.shape[1], binary_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        )
                    validate_same_shape(previous_merged, binary_mask, name_a="previous_merged", name_b="binary_mask")
                    
                    # Dilater légèrement le masque ACTUEL pour définir une zone de continuité
                    tolerance_pixels = self.analyzer.tolerance_pixel  # Ajustable si nécessaire
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance_pixels*2+1, tolerance_pixels*2+1))
                    current_dilated = cv2.dilate(binary_mask, kernel, iterations=1)
                    
                    # Ne garder du masque précédent que ce qui est PROCHE du masque actuel
                    previous_in_continuity = np.logical_and(previous_merged, current_dilated).astype(np.uint8)
                    
                    # Fusionner : masque actuel + partie continue du précédent
                    merged_mask = np.logical_or(binary_mask, previous_in_continuity).astype(np.uint8)
                else:
                    merged_mask = binary_mask.copy()
                
                
                # Prétraitement morphologique
                # Scale pixel-based parameters to keep behavior consistent when images are resized
                sf = getattr(self.analyzer, "scale_factor", 1.0) or 1.0
                closing_r = max(1, int(round(self.params['closing_radius'] * sf)))
                # Areas scale ~ sf^2
                min_size = max(1, int(round(self.params['min_object_size'] * (sf * sf))))
                
                processed = self.analyzer.preprocess_mask(
                    merged_mask,
                    closing_radius=closing_r,
                    min_size=min_size
                )
                
                # Connexion des objets (optionnel)
                if self.params['connect_objects']:
                    # Keep connection parameters consistent under resizing
                    sf = getattr(self.analyzer, "scale_factor", 1.0) or 1.0
                    lt = max(1, int(round(self.params['line_thickness'] * sf)))
                    mcd = self.params.get('max_connection_dst', None)
                    if mcd is not None:
                        mcd = max(1, int(round(mcd * sf)))
                    
                    processed = self.analyzer.merge_and_connect_objects(
                        processed,
                        line_thickness=lt,
                        max_iterations=self.params.get('max_connect_iterations', 10),
                        max_connection_distance=mcd
                    )
                
                preprocessed.append({
                    "filename": filename,
                    "day": day,
                    "modality": modality,
                    "original_mask": original_mask,
                    "processed": processed
                })
                
                # Sauvegarde pour la fusion temporelle
                previous_merged = processed.copy()
                
                # Libération mémoire partielle
                del mask, binary_mask, merged_mask
            
            
            if not preprocessed:
                self.finished.emit(pd.DataFrame())
                return
            
            # on trie au cas où par numéro de jour
            preprocessed.sort(key=lambda d: d["day"])
            
            # ----------------------------
            # 2) ORDRE CHRONOLOGIQUE : Analyser du PREMIER au DERNIER jour
            # ----------------------------
            # Ceci permet une continuité temporelle naturelle et stable
            analysis_steps = max(len(preprocessed), 1)
            analyzed_done = 0
            features_by_day = {}
            # ref_path déjà initialisé avant le preprocessing (ligne 1354)
            
            # ----------------------------
            # 3) Analyse chronologique : J0 -> J(N)
            # ----------------------------
            for item in preprocessed:
                _check_interrupt()
                day = item["day"]
                filename = item["filename"]
                
                # Progression 50 -> 100%
                self.progress.emit(
                    50 + int((analyzed_done / analysis_steps) * 50),
                    f"Analysis: {filename}"
                )
                
                # Analyser en utilisant le jour PRÉCÉDENT comme référence (continuité naturelle)
                features = self.analyzer.skeletonize_and_analyze(
                    item["processed"],
                    main_ref_path=ref_path,  # suit le jour précédent
                    grid_rows=self.params.get("grid_rows", None),
                    grid_cols=self.params.get("grid_cols", None)
                )
                
                analyzed_done += 1
                self.progress.emit(
                    50 + int((analyzed_done / analysis_steps) * 50),
                    f"Analysis: {filename}"
                )
                
                features_by_day[day] = features
                
                # Mettre à jour la référence pour le jour SUIVANT
                mp = features.get("main_path", None)
                if isinstance(mp, np.ndarray) and len(mp) >= 2:
                    ref_path = mp
                # SINON : garder l'ancienne référence (continuité malgré les trous)
                # Cela permet de "sauter" un jour problématique sans perdre la référence
            
            # Plus besoin de recalage : l'analyse chronologique assure la continuité
            
            # ----------------------------
            # 4) Construction du DataFrame + signaux Qt
            # ----------------------------
            results = []
            for item in preprocessed:
                _check_interrupt()
                day = item["day"]
                filename = item["filename"]
                modality = item["modality"]
                features = features_by_day[day]
                
                # Résumé tabulaire (sans les gros tableaux numpy)
                row = {
                    "image": filename,
                    "day": day,
                    "modality": modality,
                    **{k: v for k, v in features.items()
                       if not isinstance(v, np.ndarray)}
                }
                
                # --- Grid lengths: flatten to scalar columns for CSV/DataFrame ---
                gl = features.get("grid_lengths", None)
                if isinstance(gl, np.ndarray) and gl.ndim == 2 and gl.size > 0:
                    for i in range(gl.shape[0]):
                        for j in range(gl.shape[1]):
                            # Notation demandée: C{i,j} (1-indexed)
                            row[f"C{i+1},{j+1}"] = float(gl[i, j])
                
                gl_cm = features.get("grid_lengths_cm", None)
                if isinstance(gl_cm, np.ndarray) and gl_cm.ndim == 2 and gl_cm.size > 0:
                    for i in range(gl_cm.shape[0]):
                        for j in range(gl_cm.shape[1]):
                            row[f"C{i+1},{j+1}_cm"] = float(gl_cm[i, j])
                
                results.append(row)
                
                # Émission pour l’onglet Visualisation
                self.day_analyzed.emit(
                    day,
                    features,
                    item["original_mask"],
                    item["processed"]
                )
                
                # libération mémoire
                item["processed"] = None
                item["original_mask"] = None
            
            df = pd.DataFrame(results)
            self.finished.emit(df)
        except InterruptedError as e:
            self.error.emit(str(e))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error.emit(
                f"Error during root architecture analysis:\n{e}\n\n{tb}"
            )


class RootVisualizationWidget(QWidget):
    """Widget de visualisation du système racinaire"""
    
    def __init__(self, n_max_rows_grid=12, n_max_cols_grid=9):
        super().__init__()
        self.current_day_data = {}
        self.n_max_rows_grid = n_max_rows_grid
        self.n_max_cols_grid = n_max_cols_grid
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Canvas matplotlib
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Contrôles de visualisation
        controls = QHBoxLayout()
        
        self.show_skeleton_check = QCheckBox("Skeleton")
        self.show_skeleton_check.setChecked(True)
        self.show_skeleton_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_skeleton_check)
        
        self.show_main_path_check = QCheckBox("Main root")
        self.show_main_path_check.setChecked(True)
        self.show_main_path_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_main_path_check)
        
        self.show_secondary_check = QCheckBox("Secondary roots")
        self.show_secondary_check.setChecked(True)
        self.show_secondary_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_secondary_check)
        
        self.show_endpoints_check = QCheckBox("End points")
        self.show_endpoints_check.setChecked(True)
        self.show_endpoints_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_endpoints_check)
        
        self.show_convex_check = QCheckBox("Convex hull")
        self.show_convex_check.setChecked(True)
        self.show_convex_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_convex_check)
        
        self.show_grid_check = QCheckBox("Grid")
        self.show_grid_check.setChecked(False)
        self.show_grid_check.stateChanged.connect(self.update_visualization)
        controls.addWidget(self.show_grid_check)
        
        self.n_rows_grid_label = QLabel("X")
        controls.addWidget(self.n_rows_grid_label)
        self.n_rows_grid_combo = QComboBox()
        self.n_rows_grid_combo.addItems([str(i) for i in range(2, self.n_max_rows_grid+1)])
        self.n_rows_grid_combo.setCurrentIndex(1)
        self.n_rows_grid_combo.currentIndexChanged.connect(self.update_visualization)
        controls.addWidget(self.n_rows_grid_combo)
        
        self.n_cols_grid_label = QLabel("Y")
        controls.addWidget(self.n_cols_grid_label)
        self.n_cols_grid_combo = QComboBox()
        self.n_cols_grid_combo.addItems([str(i) for i in range(2, self.n_max_cols_grid+1)])
        self.n_cols_grid_combo.setCurrentIndex(0)
        self.n_cols_grid_combo.currentIndexChanged.connect(self.update_visualization)
        controls.addWidget(self.n_cols_grid_combo)
        
        controls.addStretch()
        layout.addLayout(controls)
        
        self.setLayout(layout)
    
    def set_day_data(self, day, features, original, merged, dataset):
        """Définir les données d'un jour spécifique"""
        self.current_day_data = {
            'day': day,
            'features': features,
            'original': original,
            'merged': merged,
            'dataset':dataset
        }
        self.update_visualization()
    
    def update_visualization(self):
        """Mise à jour de la visualisation"""
        if not self.current_day_data:
            return
        
        self.figure.clear()
        
        features = self.current_day_data['features']
        merged = self.current_day_data['merged']
        day = self.current_day_data['day']
        dataset = self.current_day_data['dataset']
        
        # Création des sous-graphiques
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # 1. Masque original
        ax1.imshow(self.current_day_data['original'], cmap='gray')
        ax1.set_title(f'Original mask - Day {day}')
        ax1.axis('off')
        
        # 2. Masque fusionné/traité
        ax2.imshow(merged, cmap='gray')
        ax2.set_title('Fused and treated mask')
        ax2.axis('off')
        
        # 3. Analyse du squelette
        if len(features.get('skeleton', [])) > 0:
            ax3.imshow(features['skeleton'], cmap='gray')
            
            if self.show_convex_check.isChecked() and len(features.get('convex_hull', [])) > 0:
                ax3.imshow(features['convex_hull'], cmap='Reds', alpha=0.2)
            
            if self.show_main_path_check.isChecked() and len(features.get('main_path', [])) > 0:
                main_path = features['main_path']
                ax3.plot(main_path[:, 1], main_path[:, 0], 'r-', linewidth=2, label='Main root')
            
            # Racines secondaires
            if self.show_secondary_check.isChecked() and len(features.get('secondary_branches', [])) > 0:
                branches = features['secondary_branches']
                first = True
                for br in np.atleast_1d(branches):
                    br = np.asarray(br)
                    if br.shape[0] < 2:
                        continue
                    if first:
                        ax3.plot(
                            br[:, 1], br[:, 0],
                            '-', linewidth=1.0, color='orange',
                            label='Secondary roots'
                        )
                        first = False
                    else:
                        ax3.plot(
                            br[:, 1], br[:, 0],
                            '-', linewidth=1.0, color='orange'
                        )
            
            if self.show_endpoints_check.isChecked() and len(features.get('endpoints', [])) > 0:
                endpoints = features['endpoints']
                ax3.scatter(endpoints[:, 1], endpoints[:, 0], c='yellow', s=30, 
                          marker='o', edgecolors='red', linewidths=1, label='End points', zorder=5)
            
            if features['centroid_x'] > 0:
                ax3.scatter(features['centroid_x_display'], features['centroid_y_display'], 
                          c='lime', s=100, marker='x', linewidths=2, label='Barycenter', zorder=5)
            
            if self.show_grid_check.isChecked():
                n_rows = int(self.n_rows_grid_combo.currentText())
                n_cols = int(self.n_cols_grid_combo.currentText())
                x_lim, y_lim = ax3.get_xlim(), ax3.get_ylim()
                x_size = x_lim[1] - x_lim[0]
                y_size = y_lim[1] - y_lim[0]
                v_lines = np.array([round(x_lim[0] + x_size / n_cols * i) for i in range(1, n_cols)], dtype='int32')
                h_lines = np.array([round(y_lim[0] + y_size / n_rows * j) for j in range(1, n_rows)], dtype='int32')
                ax3.vlines(v_lines, ymin=y_lim[0], ymax=y_lim[1], color='w', linestyles='dashed', linewidth=0.75)
                ax3.hlines(h_lines, xmin=x_lim[0], xmax=x_lim[1], color='w', linestyles='dashed', linewidth=0.75)
            
            ax3.set_title('Skeleton analysis')
            ax3.axis('off')
            ax3.legend(loc='upper right', fontsize=8)
        
        # 4. Statistiques
        ax4.axis('off')
        
        # --- Responsive rendering for the stats panel ---
        # Small screens: keep text compact, smaller font.
        # Large screens: more readable font + multi-line grid stats.
        try:
            canvas_w = int(self.canvas.width())
        except Exception:
            canvas_w = 0
        compact_view = (canvas_w > 0 and canvas_w < 1100)
        if canvas_w >= 1800:
            stats_fs = 10
        elif canvas_w >= 1400:
            stats_fs = 9
        else:
            stats_fs = 7 if compact_view else 8
        
        stats_text = f"""
        Day {day}
        Dataset: {dataset}
        
        Main root length: {features['main_root_length']:.1f} px
        Secondary roots length: {features['secondary_roots_length']:.1f} px
        Total length (graph): {features['total_root_length']:.1f} px
        Exact skeleton length: {features.get('exact_skeleton_length', 0):.1f} px
        
        Number of branches: {features['branch_count']}
        Number of end points without pruning: {features['endpoint_count_raw']}
        Number of roots without pruning: {features['root_count_raw']}
        Number of end points: {features['endpoint_count']}
        Number of roots: {features['root_count']}
        Root count attach: {features['root_count_attach']}
        
        Mean secondary angles: {features['mean_secondary_angles']:.2f}
        Standard deviation secondary angles: {features['std_secondary_angles']:.2f}
        Mean absolute secondary angles: {features['mean_abs_secondary_angles']:.2f}
        Std absolute secondary angles: {features['std_abs_secondary_angles']:.2f}
        
        Total area: {features['total_area']:.0f} px²
        Convex area: {features['convex_area']:.0f} px²
        
        Barycenter: ({features['centroid_x']:.1f}, {features['centroid_y']:.1f})
        """
        
        # --- Grid: add a short summary line (heatmap moved to Heatmap tab) ---
        gl = features.get("grid_lengths", None)
        if isinstance(gl, np.ndarray) and gl.ndim == 2 and gl.size > 0:
            try:
                gmin, gmean, gmax = float(np.min(gl)), float(np.mean(gl)), float(np.max(gl))
                if compact_view:
                    stats_text += f"""
        Grid ({gl.shape[0]}x{gl.shape[1]}): {gmin:.2f}/{gmean:.2f}/{gmax:.2f} px
        """
                else:
                    stats_text += f"""
        Grid per cell ({gl.shape[0]}x{gl.shape[1]}):
            - Min:  {gmin:.2f} px
            - Mean: {gmean:.2f} px
            - Max:  {gmax:.2f} px
        """
            except Exception:
                pass
        
        # Background filling the whole panel (so it's not a tiny centered box)
        try:
            bg = FancyBboxPatch(
                (0.0, 0.0), 1.0, 1.0,
                transform=ax4.transAxes,
                boxstyle="round,pad=0.02",
                facecolor="wheat",
                edgecolor="none",
                alpha=0.35,
                zorder=0,
            )
            ax4.add_patch(bg)
        except Exception:
            pass
        
        ax4.text(
            0.03, 0.97, stats_text,
            transform=ax4.transAxes,
            fontsize=stats_fs,
            verticalalignment='top',
            horizontalalignment='left',
            fontfamily='monospace',
            wrap=True,
            zorder=1,
        )
        
        self.figure.tight_layout()
        self.canvas.draw()

class RootHeatmapWidget(QWidget):
    """Affiche la heatmap des longueurs par cellule de grille (grid_lengths / grid_lengths_cm)."""
    
    def __init__(self):
        super().__init__()
        self.current = None  # dict(day, features, dataset)
        
        layout = QVBoxLayout(self)
        
        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Unité:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["px", "cm"])
        self.unit_combo.setCurrentIndex(0)
        self.unit_combo.currentIndexChanged.connect(self.update_view)
        ctrl.addWidget(self.unit_combo)
        
        ctrl.addSpacing(20)
        
        ctrl.addWidget(QLabel("Palette:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "YlOrRd",      # Jaune -> Orange -> Rouge
            "viridis",     # Bleu -> Vert -> Jaune
            "plasma",      # Violet -> Rose -> Jaune
            "inferno",     # Noir -> Violet -> Jaune
            "Blues",       # Blanc -> Bleu foncé
            "Greens",      # Blanc -> Vert foncé
            "Reds",        # Blanc -> Rouge foncé
            "YlGn",        # Jaune -> Vert
            "RdYlGn",      # Rouge -> Jaune -> Vert
            "coolwarm",    # Bleu -> Blanc -> Rouge
        ])
        self.cmap_combo.setCurrentIndex(0)  # YlOrRd par défaut
        self.cmap_combo.currentIndexChanged.connect(self.update_view)
        ctrl.addWidget(self.cmap_combo)
        
        ctrl.addSpacing(20)
        
        # Option pour inverser la colormap
        self.invert_cmap_check = QCheckBox("Inverser")
        self.invert_cmap_check.stateChanged.connect(self.update_view)
        ctrl.addWidget(self.invert_cmap_check)
        
        ctrl.addStretch()
        layout.addLayout(ctrl)
        
        # Figure
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def set_day_data(self, day, features, dataset=None):
        self.current = {"day": day, "features": features, "dataset": dataset}
        self.update_view()
    
    def update_view(self):
        if not self.current:
            return
        features = self.current["features"]
        day = self.current["day"]
        dataset = self.current.get("dataset", "")
        
        use_cm = (self.unit_combo.currentText().lower() == "cm")
        key = "grid_lengths_cm" if use_cm else "grid_lengths"
        gl = features.get(key, None)
        
        # Récupérer la colormap sélectionnée
        cmap_name = self.cmap_combo.currentText()
        if self.invert_cmap_check.isChecked():
            cmap_name += "_r"  # Version inversée
        
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)
        
        if isinstance(gl, np.ndarray) and gl.ndim == 2 and gl.size > 0:
            # Afficher la heatmap avec la colormap sélectionnée
            im = ax.imshow(gl, aspect="auto", origin="upper", cmap=cmap_name, interpolation="nearest")
            
            # Ajouter une barre de légende (colorbar)
            cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Longueur ({'cm' if use_cm else 'px'})", rotation=270, labelpad=15)
            
            ax.set_title(f"Grid length heatmap ({'cm' if use_cm else 'px'}) - J{day} - {dataset}")
            ax.set_xticks(range(gl.shape[1]))
            ax.set_yticks(range(gl.shape[0]))
            ax.set_xlabel("col")
            ax.set_ylabel("row")
            
            # Annoter chaque cellule (lisible tant que petite grille)
            try:
                if gl.shape[0] <= 12 and gl.shape[1] <= 12:
                    # Déterminer la couleur du texte selon la luminosité de fond
                    vmin = gl.min()
                    vmax = gl.max()
                    threshold = vmin + (vmax - vmin) * 0.5  # seuil à 50% de la plage
                    
                    for i in range(gl.shape[0]):
                        for j in range(gl.shape[1]):
                            # Texte blanc sur fond foncé, noir sur fond clair
                            text_color = 'white' if gl[i, j] > threshold else 'black'
                            ax.text(j, i, f"{gl[i, j]:.1f}", 
                                   ha="center", va="center", 
                                   fontsize=8, 
                                   color=text_color,
                                   weight='bold')
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, "Aucune donnée de grille disponible pour ce jour.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        
        self.figure.tight_layout()
        self.canvas.draw()


class RootArchitectureWindow(QMainWindow):
    """Fenêtre optimisée avec paramètres de performance"""
    
    def __init__(self, parent=None, init_params=None, datasets=None, current_dir=None, output_dir=None, screen_size=None):
        super().__init__(parent)
        self.current_df = None
        self.daily_data = {}
        self.worker = None
        
        self._busy_cm = None
        self.default_params  =  { 'closing_radius': 5,
                                  'closing_shape': cv2.MORPH_RECT,
                                  'min_branch_length': 20.0,
                                  'min_connected_components_area': 0,
                                  'min_object_size': 200,
                                  'max_connection_dst': 240,
                                  'line_thickness': 5,
                                  'temporal_merge': True,
                                  'connect_objects': True,
                                  'main_path_bias': 20,
                                  'grid_rows': 3,
                                  'grid_cols': 2
                                }
        self.init_params = init_params
        self.datasets = datasets
        self.datasets_list = ["No selection"]
        self.current_dataset = None
        self.current_directory = current_dir
        self.output_directory = output_dir
        self.screen_size = screen_size
        self.max_analysis_window_size = [2400, 1800]
        self.analysis_geometry = None
        self.image_extension_list = tuple([os.extsep + ext for ext in ['png', 'jpg', 'tif', 'tiff']])
        
        self.n_max_rows_grid = 12
        self.n_max_cols_grid = 9
        
        # Default closing parameters
        self.kernel_shape_dict = {"Rectangle":cv2.MORPH_RECT, "Cross":cv2.MORPH_CROSS, "Ellipse":cv2.MORPH_ELLIPSE}
        self.default_kernel_shape_index = 0
        self.default_kernel_shape_name = list(self.kernel_shape_dict.keys())[self.default_kernel_shape_index]
        
        self.kernel_shape_index = self.default_kernel_shape_index
        self.kernel_shape = self.default_kernel_shape_name
        self.kernel_shape_value = self.kernel_shape_dict[self.default_kernel_shape_name]
        
        self.init_parameters()
        self.init_ui()
    
    def init_parameters(self):
        if self.init_params is None:
            self.init_params =  self.default_params
        else:
            for param_key, key_value in self.default_params.items():
                self.init_params[param_key] = self.init_params.get(param_key, key_value)
        
        self.kernel_shape_value = self.init_params['closing_shape']
        for idx_key, key_value in enumerate(self.kernel_shape_dict.items()):
            if key_value == self.kernel_shape_value:
                self.kernel_shape_index = idx_key
                self.kernel_shape = list(self.kernel_shape_dict.keys())[idx_key]
                break
        
        if self.datasets:
            self.datasets_list += list(self.datasets.keys())
        
        # Dataset selector defaults
        self._show_only_segmented = True
        # Ajoute un attribut .selected si absent (pour le widget)
        if self.datasets:
            for _name, _ds in self.datasets.items():
                if not hasattr(_ds, 'selected'):
                    try:
                        _ds.selected = self._dataset_has_root_masks(_ds)
                    except Exception:
                        _ds.selected = False
        
        if self.screen_size is not None:
            self.analysis_geometry = [0, 0, min(int(self.screen_size[0] * 0.9), self.max_analysis_window_size[0]), min(int(self.screen_size[1] * 0.9), self.max_analysis_window_size[1])]
            self.analysis_geometry[0] = min(100, self.screen_size[0] - self.analysis_geometry[2])
            self.analysis_geometry[1] = min(100, self.screen_size[1] - self.analysis_geometry[3])
        
        
        # Batch state
        self._batch_active = False
        self._batch_all_results = []
        self._batch_queue = []
        self._batch_index = 0
        self._batch_current = None
        self._batch_running = False
        self._batch_total = 0
        self._stop_requested = False
        # Cache results per dataset so switching datasets refreshes the visuals instantly
        self._results_cache = {}
        
        self._analysis_dataset = None
    
    def init_ui(self):
        self.setWindowTitle("Root Architectural Analysis")
        
        # --- Taille initiale adaptative ---
        try:
            if self.screen_size is None:
                screen = QApplication.primaryScreen()
                geom = screen.availableGeometry() if screen is not None else None
                sw = int(geom.width()) if geom is not None else 1600
                sh = int(geom.height()) if geom is not None else 900
                
                win_w = min(1600, int(sw * 0.95))
                win_h = min(900,  int(sh * 0.90))
                self.setGeometry(50, 50, win_w, win_h)
            else:
                self.setGeometry(self.analysis_geometry[0], self.analysis_geometry[1], self.analysis_geometry[2], self.analysis_geometry[3])
                sh = self.analysis_geometry[3]
                
            # Réduction légère de la police pour les faibles résolutions et garder l'UI lisible
            if sh <= 900:
                f = self.font()
                if f.pointSize() > 0:
                    f.setPointSize(max(8, f.pointSize() - 1))
                    self.setFont(f)
        except Exception:
            self.setGeometry(100, 100, 1600, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        
        # === PANNEAU GAUCHE ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Fichiers
        file_group = QGroupBox("Root mask files")
        file_layout = QVBoxLayout()
        
        self.select_btn = QPushButton("Select masks")
        self.select_btn.clicked.connect(self.select_mask_files)
        file_layout.addWidget(self.select_btn)
        
        self.files_label = QLabel("No files selected")
        self.files_label.setWordWrap(True)
        file_layout.addWidget(self.files_label)
        
        # --- Dataset selector (remplace la QComboBox) ---
        self.dataset_selection_label = QLabel("Dataset selection:")
        self.dataset_selection_label.setStyleSheet("font-weight: bold; font-size: 13px")
        file_layout.addWidget(self.dataset_selection_label)
        
        self.only_segmented_check = QCheckBox("Show only datasets with segmented roots")
        self.only_segmented_check.setChecked(True)
        self.only_segmented_check.stateChanged.connect(self._on_only_segmented_toggled)
        file_layout.addWidget(self.only_segmented_check)
        
        self.dataset_selector = DatasetSelectionWidget(parent=self, title="")
        self.dataset_selector.dataset_view_requested.connect(self._on_dataset_view_requested)
        
        # On s'assure que le choix des datasets soit toujours utilisable sur de faibles résolutions
        try:
            self.dataset_selector.list_widget.setMinimumHeight(120)
        except Exception:
            pass
        
        file_layout.addWidget(self.dataset_selector)
        
        self.refresh_dataset_selector(keep_view=False)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # === PARAMÈTRES D'OPTIMISATION ===
        optim_group = QGroupBox("⚡ Performance Settings")
        optim_layout = QVBoxLayout()
        
        # Redimensionnement et échantillonnage
        resize_layout = QGridLayout()
        resize_layout.addWidget(QLabel("Maximum image size:"), 0, 0)
        self.resize_spin = QSpinBox()
        self.resize_spin.setRange(0, 10000)
        self.resize_spin.setValue(2000)
        self.resize_spin.setSingleStep(500)
        self.resize_spin.setSpecialValueText("None (low)")
        resize_layout.addWidget(self.resize_spin, 0, 1)
        
        resize_layout.addWidget(QLabel("Maximum pixel before sampling:"), 1, 0)
        self.sampling_threshold_spin = QSpinBox()
        self.sampling_threshold_spin.setRange(10, 100000000)
        self.sampling_threshold_spin.setValue(100000)
        self.sampling_threshold_spin.setSingleStep(10000)
        resize_layout.addWidget(self.sampling_threshold_spin, 1, 1)
        
        optim_layout.addLayout(resize_layout)
        
        info_label = QLabel("💡 2000px recommended for speed/accuracy balance")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        info_label.setWordWrap(True)
        optim_layout.addWidget(info_label)
        
        # Taux d'échantillonnage
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("Sampling:"))
        self.sample_spin = QDoubleSpinBox()
        self.sample_spin.setRange(0.01, 1.00)
        self.sample_spin.setValue(1.00)
        self.sample_spin.setSingleStep(0.01)
        self.sample_spin.setDecimals(2)
        sample_layout.addWidget(self.sample_spin)
        optim_layout.addLayout(sample_layout)
        
        # Itérations max connexion
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Maximum iterations:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(1, 100)
        self.iter_spin.setValue(10)
        iter_layout.addWidget(self.iter_spin)
        optim_layout.addLayout(iter_layout)
        
        optim_group.setLayout(optim_layout)
        left_layout.addWidget(optim_group)
        
        # Paramètres standard
        params_group = QGroupBox("Analysis parameters")
        params_layout = QVBoxLayout()
        
        closing_layout = QHBoxLayout()
        closing_layout.addWidget(QLabel("Closing radius:"))
        self.closing_spin = QSpinBox()
        self.closing_spin.setRange(1, 200)
        self.closing_spin.setValue(5)
        closing_layout.addWidget(self.closing_spin)
        params_layout.addLayout(closing_layout)
        
        min_branch_layout = QHBoxLayout()
        min_branch_layout.addWidget(QLabel("Minimum branch size:"))
        self.min_branch_length_spin = QDoubleSpinBox()
        self.min_branch_length_spin.setRange(0.0, 100000.0)
        self.min_branch_length_spin.setValue(20.0)
        self.min_branch_length_spin.setDecimals(2)
        min_branch_layout.addWidget(self.min_branch_length_spin)
        params_layout.addLayout(min_branch_layout)
        
        min_object_size_layout = QHBoxLayout()
        min_object_size_layout.addWidget(QLabel("Minimum object size:"))
        self.min_object_size_spin = QSpinBox()
        self.min_object_size_spin.setRange(0, 100000)
        self.min_object_size_spin.setValue(200)
        min_object_size_layout.addWidget(self.min_object_size_spin)
        params_layout.addLayout(min_object_size_layout)
        
        max_connection_dst_layout = QHBoxLayout()
        max_connection_dst_layout.addWidget(QLabel("Maximum connection distance:"))
        self.max_connection_dst_spin = QSpinBox()
        self.max_connection_dst_spin.setRange(0, 2000)
        self.max_connection_dst_spin.setValue(240)
        max_connection_dst_layout.addWidget(self.max_connection_dst_spin)
        params_layout.addLayout(max_connection_dst_layout)
        
        thickness_layout = QHBoxLayout()
        thickness_layout.addWidget(QLabel("Connection thickness:"))
        self.thickness_spin = QSpinBox()
        self.thickness_spin.setRange(1, 200)
        self.thickness_spin.setValue(5)
        thickness_layout.addWidget(self.thickness_spin)
        params_layout.addLayout(thickness_layout)
        
        main_path_bias_layout = QHBoxLayout()
        main_path_bias_layout.addWidget(QLabel("Main path bias:"))
        self.main_path_bias_spin = QSpinBox()
        self.main_path_bias_spin.setRange(0, 1000000)
        self.main_path_bias_spin.setValue(20)
        self.main_path_bias_spin.setSingleStep(1)
        main_path_bias_layout.addWidget(self.main_path_bias_spin)
        params_layout.addLayout(main_path_bias_layout)
        
        tolerance_pixel_layout = QHBoxLayout()
        tolerance_pixel_layout.addWidget(QLabel("Fusion tolerance pixel:"))
        self.tolerance_pixel_spin = QSpinBox()
        self.tolerance_pixel_spin.setRange(0, 200)
        self.tolerance_pixel_spin.setValue(5)
        self.tolerance_pixel_spin.setSingleStep(1)
        tolerance_pixel_layout.addWidget(self.tolerance_pixel_spin)
        params_layout.addLayout(tolerance_pixel_layout)
        
        pixels_per_cm_layout = QHBoxLayout()
        pixels_per_cm_layout.addWidget(QLabel("Pixels/cm"))
        self.pixels_per_cm_spin = QDoubleSpinBox()
        self.pixels_per_cm_spin.setRange(0.0, 10000.0)
        self.pixels_per_cm_spin.setValue(0.0)
        self.pixels_per_cm_spin.setDecimals(3)
        pixels_per_cm_layout.addWidget(self.pixels_per_cm_spin)
        params_layout.addLayout(pixels_per_cm_layout)
        
        additional_layout = QHBoxLayout()
        checkbox_layout = QVBoxLayout()
        self.temporal_check = QCheckBox("Temporal fusion")
        self.temporal_check.setChecked(True)
        checkbox_layout.addWidget(self.temporal_check)
        
        self.connect_check = QCheckBox("Connect objects")
        self.connect_check.setChecked(True)
        checkbox_layout.addWidget(self.connect_check)
        
        grid_combobox_layout = QVBoxLayout()
        grid_title = QLabel("Grid rows and columns:")
        self.n_grid_rows_combobox = QComboBox()
        self.n_grid_rows_combobox.addItems([str(i) for i in range(2, self.n_max_rows_grid+1)])
        self.n_grid_rows_combobox.setCurrentIndex(1)
        self.n_grid_cols_combobox = QComboBox()
        self.n_grid_cols_combobox.addItems([str(i) for i in range(2, self.n_max_cols_grid+1)])
        self.n_grid_cols_combobox.setCurrentIndex(0)
        grid_combobox_layout.addWidget(grid_title)
        grid_combobox_layout.addWidget(self.n_grid_rows_combobox)
        grid_combobox_layout.addWidget(self.n_grid_cols_combobox)
        additional_layout.addLayout(checkbox_layout)
        additional_layout.addLayout(grid_combobox_layout)
        params_layout.addLayout(additional_layout)
        
        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)
        
        # Exécution
        exec_group = QGroupBox("Execute")
        exec_layout = QVBoxLayout()
        
        self.run_btn = QPushButton("Analyze current dataset")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        exec_layout.addWidget(self.run_btn)
        
        self.batch_btn = QPushButton("Analyze selected datasets")
        self.batch_btn.clicked.connect(self.batch_process_all_datasets)
        self.batch_btn.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 12px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #2196F3;
            }
        """)
        exec_layout.addWidget(self.batch_btn)
        
        self.stop_btn = QPushButton("⛔ Stop")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #E53935;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 11pt;
            }
            
            QPushButton:hover {
                background-color: #D32F2F;
            }
            
            QPushButton:disabled {
                background-color: #BDBDBD;
                color: #666666;
            }
        """)
        exec_layout.addWidget(self.stop_btn)
        
        self.progress_bar = QProgressBar()
        exec_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        exec_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        exec_layout.addWidget(self.log_text)
        
        exec_group.setLayout(exec_layout)
        left_layout.addWidget(exec_group)
        
        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_csv_btn = QPushButton("Exporter results (CSV)")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setEnabled(False)
        export_layout.addWidget(self.export_csv_btn)
        
        self.export_images_btn = QPushButton("Export visualizations")
        self.export_images_btn.clicked.connect(self.export_visualizations)
        self.export_images_btn.setEnabled(False)
        export_layout.addWidget(self.export_images_btn)
        
        export_group.setLayout(export_layout)
        left_layout.addWidget(export_group)
        
        left_layout.addStretch()
        
        # On ajoute un scrolling verticale au panneau de gauche
        # Cela prévient l'écrasement de la liste des datasets sur de faibles résolutions
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        try:
            left_scroll.setFrameShape(QScrollArea.Shape.NoFrame if PYQT_VERSION == 6 else QScrollArea.NoFrame)
        except Exception:
            pass
        left_scroll.setWidget(left_panel)
        
        # --- Anti "roulette change les paramètres" ---
        # Si l'utilisateur scrolle dans le panneau de gauche, on veut scroller la QScrollArea,
        # pas changer les valeurs des SpinBox/ComboBox sous le curseur.
        self._left_wheel_filter = WheelToScrollAreaFilter(left_scroll, self)
        focus_policy = Qt.FocusPolicy.StrongFocus if hasattr(Qt, "FocusPolicy") else Qt.StrongFocus
        for w in left_panel.findChildren((QSpinBox, QDoubleSpinBox, QComboBox)):
            try:
                w.setFocusPolicy(focus_policy)
                w.installEventFilter(self._left_wheel_filter)
            except Exception:
                pass
        
        # On évite le scrolling horizontal (on préfère wrapping + largeur fixe)
        try:
            left_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff if PYQT_VERSION == 6 else Qt.ScrollBarAlwaysOff
            )
        except Exception:
            pass
        
        splitter.addWidget(left_scroll)
        
        # === PANNEAU DROIT (identique) ===
        right_panel = QTabWidget()
        
        # Visualisation
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        day_control = QHBoxLayout()
        day_control.setContentsMargins(0, 0, 0, 0)
        day_control.setSpacing(6)
        day_control.addWidget(QLabel("Day:"))
        
        self.day_slider = QSlider(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.day_slider.setMinimum(0)
        self.day_slider.setMaximum(0)
        self.day_slider.setFixedHeight(18)
        self.day_slider.valueChanged.connect(self.update_day_visualization)
        day_control.addWidget(self.day_slider)
        
        self.day_label = QLabel("0")
        day_control.addWidget(self.day_label)
        
        # conteneur bloqué en hauteur
        day_container = QWidget()
        day_container.setLayout(day_control)
        day_container.setFixedHeight(26)
        
        size_policy = day_container.sizePolicy()
        size_policy.setVerticalPolicy(QSizePolicy.Policy.Fixed if PYQT_VERSION == 6 else QSizePolicy.Fixed)
        day_container.setSizePolicy(size_policy)
        
        viz_layout.addWidget(day_container)
        
        self.viz_widget = RootVisualizationWidget(n_max_rows_grid=self.n_max_rows_grid, n_max_cols_grid=self.n_max_cols_grid)
        viz_layout.addWidget(self.viz_widget)
        
        right_panel.addTab(viz_tab, "Visualisation")
        
        # Table
        self.results_table = QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        right_panel.addTab(self.results_table, "Results")
        
        # Graphiques
        plots_tab = QWidget()
        plots_layout = QVBoxLayout(plots_tab)
        
        plot_control = QHBoxLayout()
        plot_control.addWidget(QLabel("Variable:"))
        self.plot_combo = QComboBox()
        self.plot_combo.addItems([
            'main_root_length', 'secondary_roots_length', 'total_root_length',
            'secondary_roots_length', 'exact_skeleton_length', 'branch_count',
            'endpoint_count_raw', 'root_count_raw', 'endpoint_count', 'root_count',
            'root_count_attach', 'root_count_raw_cum', 'root_count_cum', 'root_count_attach_cum', 
            'total_area', 'convex_hull', 'convex_area', 'mean_secondary_angles', 
            'mean_abs_secondary_angles', 'std_secondary_angles', 'std_abs_secondary_angles'
        ])
        self.plot_combo.currentTextChanged.connect(self.update_plot)
        plot_control.addWidget(self.plot_combo)
        plot_control.addStretch()
        plots_layout.addLayout(plot_control)
        
        self.plot_figure = Figure(figsize=(10, 6))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        plots_layout.addWidget(self.plot_canvas)
        
        right_panel.addTab(plots_tab, "Graphs")
        
        # Heatmap
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        
        # Day slider (Heatmap tab) - kept independent from the Visualization tab checkbox logic.
        heatmap_day_row = QWidget()
        heatmap_day_control = QHBoxLayout(heatmap_day_row)
        heatmap_day_control.setContentsMargins(0, 0, 0, 0)
        heatmap_day_control.setSpacing(8)
        heatmap_day_control.addWidget(QLabel("Day:"))
        self.heatmap_day_slider = QSlider(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        self.heatmap_day_slider.setMinimum(0)
        self.heatmap_day_slider.setMaximum(0)
        # prevent vertical growth
        try:
            from PyQt5.QtWidgets import QSizePolicy as _QSizePolicy5
            _QSizePolicy = _QSizePolicy5
        except Exception:
            try:
                from PyQt6.QtWidgets import QSizePolicy as _QSizePolicy6
                _QSizePolicy = _QSizePolicy6
            except Exception:
                _QSizePolicy = None
        if _QSizePolicy is not None:
            if hasattr(_QSizePolicy, 'Policy'):
                self.heatmap_day_slider.setSizePolicy(_QSizePolicy.Policy.Expanding, _QSizePolicy.Policy.Fixed)
            else:
                self.heatmap_day_slider.setSizePolicy(_QSizePolicy.Expanding, _QSizePolicy.Fixed)
        self.heatmap_day_slider.setFixedHeight(18)
        # Also constrain the whole row so it doesn't steal vertical space
        heatmap_day_row.setFixedHeight(26)
        self.heatmap_day_slider.valueChanged.connect(self._on_heatmap_day_changed)
        heatmap_day_control.addWidget(self.heatmap_day_slider)
        self.heatmap_day_label = QLabel("0")
        heatmap_day_control.addWidget(self.heatmap_day_label)
        heatmap_layout.addWidget(heatmap_day_row)
        
        self.heatmap_widget = RootHeatmapWidget()
        heatmap_layout.addWidget(self.heatmap_widget)
        try:
            heatmap_layout.setStretch(0, 0)
            heatmap_layout.setStretch(1, 1)
        except Exception:
            pass
        right_panel.addTab(heatmap_tab, "Heatmap")
        
        splitter.addWidget(right_panel)
        splitter.setSizes([450, 1150])
        
        main_layout.addWidget(splitter)
        self.mask_files = []
        
        # Après construction complète de l'UI, sélectionner un dataset par défaut (view)
        QTimer.singleShot(0, lambda: self.refresh_dataset_selector(keep_view=False))
    
    
    # -------------------------
    # Gestion plus sûre des widgets
    # -------------------------
    def _safe_set_text(self, attr, text):
        w = getattr(self, attr, None)
        if w is not None:
            w.setText(text)
    
    def _safe_set_enabled(self, attr, enabled):
        w = getattr(self, attr, None)
        if w is not None:
            w.setEnabled(enabled)
    
    def _safe_set_value(self, attr, text):
        w = getattr(self, attr, None)
        if w is not None:
            w.setValue(text)
    
    # -------------------------
    # Dataset helpers (selection widget)
    # -------------------------
    def _dataset_has_root_masks(self, ds):
        """True si le dataset a un dossier 'roots' avec au moins un masque image."""
        if ds is None:
            return False
        try:
            seg_dir = ds.segmented_directory.get('roots', None)
        except Exception:
            return False
        if not seg_dir or not os.path.isdir(seg_dir):
            return False
        files = [f for f in os.listdir(seg_dir) if f.lower().endswith(self.image_extension_list)]
        return len(files) > 0
    
    def _get_datasets_for_selector(self):
        """Retourne le dict de datasets à afficher dans le widget."""
        if not self.datasets:
            return {}
        if getattr(self, "_show_only_segmented", True):
            return {k: v for k, v in self.datasets.items() if self._dataset_has_root_masks(v)}
        return dict(self.datasets)
    
    def refresh_dataset_selector(self, keep_view=True):
        d = self._get_datasets_for_selector()
        
        ui_ready = hasattr(self, "run_btn")  # run_btn est créé tard dans init_ui()
        current_view = self.dataset_selector.current_viewing_dataset() if keep_view else None
        
        # Tant que l'UI n'est pas prête, on ne sélectionne PAS automatiquement un dataset à "view"
        self.dataset_selector.set_datasets(d, select_first_for_view=ui_ready)
        
        if ui_ready and current_view and current_view in d:
            self.dataset_selector.set_viewing_dataset(current_view, emit_signal=False)
            self.change_dataset_files(current_view)
    
    def _on_only_segmented_toggled(self, state):
        self._show_only_segmented = (state == (Qt.CheckState.Checked if PYQT_VERSION == 6 else Qt.Checked))
        self.refresh_dataset_selector(keep_view=True)
    
    def _on_dataset_view_requested(self, dataset_name):
        self.change_dataset_files(dataset_name)
    
    def select_mask_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select masks", self.output_directory, 
            "Images (*.png *.jpg *.tiff *.tif *.bmp)"
        )
        
        if files:
            self.mask_files = sorted(files)
            mask_dir = os.path.basename(self.mask_files[0])
            self.current_dataset = os.path.split(mask_dir)[-2]
            self.files_label.setText(f"{len(files)} selected files")
            self._safe_set_enabled("run_btn", True)
            self.log(f"✓ {len(files)} files")
            
            days = []
            for f in self.mask_files:
                day_match = re.search(r"_J(\d+)_", Path(f).name)
                if day_match:
                    days.append(int(day_match.group(1)))
                else:
                    day_match = re.search(r"_D(\d+)_", Path(f).name)
                    if day_match:
                        days.append(int(day_match.group(1)))
            if days:
                self.day_slider.setMinimum(min(days))
                self.day_slider.setMaximum(max(days))
                self.day_label.setText(str(min(days)))
    
    def change_dataset_files(self, dataset_name=None):
        """Charge les fichiers (masques racines) du dataset demandé et met à jour l'UI."""
        if not self.datasets:
            self.current_dataset = None
            self.mask_files = []
            self.files_label.setText("0 selected files")
            self._safe_set_enabled("run_btn", False)
            return
        
        if dataset_name is None:
            dataset_name = self.current_dataset
        
        if not dataset_name or dataset_name not in self.datasets:
            self.current_dataset = None
            self.mask_files = []
            self._safe_set_text("files_label", "0 selected files")
            self._safe_set_enabled("run_btn", False)
            return
        
        self.current_dataset = dataset_name
        
        ds = self.datasets[self.current_dataset]
        try:
            seg_dir = ds.segmented_directory.get('roots', None)
        except Exception:
            seg_dir = None
        
        if not seg_dir or not os.path.isdir(seg_dir):
            self.mask_files = []
            self._safe_set_text("files_label", "0 selected files")
            self._safe_set_enabled("run_btn", False)
            self.log(f"✗ No segmented roots directory for: {self.current_dataset}")
            return
        
        self.mask_files = sorted([
            os.path.join(seg_dir, f)
            for f in os.listdir(seg_dir)
            if f.lower().endswith(self.image_extension_list)
        ])
        
        self._safe_set_text("files_label", f"{len(self.mask_files)} selected files")
        self._safe_set_enabled("run_btn", len(self.mask_files) > 0)
        self.log(f"✓ {len(self.mask_files)} files")
        
        # If this dataset has been analyzed already during this session, restore cached results
        cache = None
        try:
            cache = getattr(self, "_results_cache", {}).get(self.current_dataset, None)
        except Exception:
            cache = None
        
        if cache is not None:
            try:
                self.current_df = cache.get("df", None)
                self.daily_data = cache.get("daily_data", {}) or {}
            except Exception:
                self.current_df = None
                self.daily_data = {}
            
            # Refresh Results / Graphs / Visualisation immediately from cached data
            try:
                if self.current_df is not None:
                    self.populate_results_table(self.current_df)
                    self.update_plot()
            except Exception:
                pass
            
            # Sync sliders from cached days (important when user switches datasets after analysis)
            try:
                if self.daily_data:
                    days = sorted(self.daily_data.keys())
                    dmin, dmax = min(days), max(days)
                    # When switching dataset for visualization, always reset to the first available day
                    try:
                        self.day_slider.blockSignals(True)
                        self.day_slider.setMinimum(dmin)
                        self.day_slider.setMaximum(dmax)
                        self.day_slider.setValue(dmin)
                    finally:
                        self.day_slider.blockSignals(False)
                    self.day_label.setText(str(dmin))
                    # Heatmap slider if present
                    if hasattr(self, "heatmap_day_slider") and self.heatmap_day_slider is not None:
                        try:
                            self.heatmap_day_slider.blockSignals(True)
                            self.heatmap_day_slider.setMinimum(dmin)
                            self.heatmap_day_slider.setMaximum(dmax)
                            self.heatmap_day_slider.setValue(dmin)
                        finally:
                            self.heatmap_day_slider.blockSignals(False)
                        if hasattr(self, "heatmap_day_label") and self.heatmap_day_label is not None:
                            self.heatmap_day_label.setText(str(dmin))
                    self.update_day_visualization()
            except Exception:
                pass
            
            # Buttons
            try:
                self._safe_set_enabled("export_csv_btn", self.current_df is not None and not self.current_df.empty)
                self._safe_set_enabled("export_images_btn", True)
            except Exception:
                pass
            return
        
        # No cache: clear current analysis state
        self.daily_data = {}
        self.current_df = None
        
        days = []
        for f in self.mask_files:
            name = Path(f).name
            m = re.search(r"_J(\d+)_", name)
            if m:
                days.append(int(m.group(1)))
                continue
            m = re.search(r"_D(\d+)_", name)
            if m:
                days.append(int(m.group(1)))
        
        if days:
            dmin, dmax = min(days), max(days)
            try:
                self.day_slider.blockSignals(True)
                self.day_slider.setMinimum(dmin)
                self.day_slider.setMaximum(dmax)
                self.day_slider.setValue(dmin)
            finally:
                self.day_slider.blockSignals(False)
            self.day_label.setText(str(dmin))
            # Keep Heatmap slider/label consistent even when there is no cached analysis yet.
            if hasattr(self, "heatmap_day_slider") and self.heatmap_day_slider is not None:
                try:
                    self.heatmap_day_slider.blockSignals(True)
                    self.heatmap_day_slider.setMinimum(dmin)
                    self.heatmap_day_slider.setMaximum(dmax)
                    self.heatmap_day_slider.setValue(dmin)
                finally:
                    self.heatmap_day_slider.blockSignals(False)
                if hasattr(self, "heatmap_day_label") and self.heatmap_day_label is not None:
                    self.heatmap_day_label.setText(str(dmin))
            self.update_day_visualization()
        else:
            self.day_slider.setMinimum(0)
            self.day_slider.setMaximum(0)
            self.day_slider.setValue(0)
            self.day_label.setText("0")
    
    def get_analysis_params(self):
        """Récupération des paramètres incluant les paramètres d'optimisations """
        return {
            'closing_radius': self.closing_spin.value(),
            'min_branch_length': self.min_branch_length_spin.value(),
            'min_object_size': self.min_object_size_spin.value(),
            'line_thickness': self.thickness_spin.value(),
            'temporal_merge': self.temporal_check.isChecked(),
            'max_connection_dst':self.max_connection_dst_spin.value(),
            'connect_objects': self.connect_check.isChecked(),
            'main_path_bias': self.main_path_bias_spin.value(),
            'fusion_tolerance_pixel': self.tolerance_pixel_spin.value(),
            'pixels_per_cm': self.pixels_per_cm_spin.value(),
            'grid_rows': int(self.n_grid_rows_combobox.currentText()),
            'grid_cols': int(self.n_grid_cols_combobox.currentText()),
            # Paramètres d'optimisation
            'max_image_size': self.resize_spin.value(),
            'min_sampling_threshold': self.sampling_threshold_spin.value(),
            'connection_sample_rate': self.sample_spin.value(),
            'max_connect_iterations': self.iter_spin.value()
        }
    
    def get_segmented_root_datasets(self):
        valid_datasets = []
        for name, info in self.datasets.items():
            seg_dir = info.segmented_directory.get('roots', None)
            if seg_dir and os.path.exists(seg_dir):
                images = [
                    f for f in os.listdir(seg_dir)
                    if f.lower().endswith(self.image_extension_list)
                ]
                if len(images) > 0:
                    valid_datasets.append(name)
        return valid_datasets
    
    def get_dataset_index(self, dataset_name):
        if dataset_name not in self.datasets_list:
            return
        
        for idx, dataset in enumerate(self.datasets_list):
            if dataset_name.lower() == dataset.lower():
                return idx
    
    def stop_analysis(self):
        """Arrête l'analyse en cours (single or batch)  """
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            return
        
        # Stop batch chaining immediately (prevents launching the next dataset)
        if getattr(self, "_batch_active", False):
            self._batch_active = False
            try:
                self._batch_queue = []
            except Exception:
                pass
        
        self._stop_requested = True
        self.log("⛔ Stop requested...")
        
        # UI feedback
        self._safe_set_enabled("stop_btn", False)
        self._safe_set_text("status_label", "Stopping...")
        
        # Ask the worker to stop (no terminate(), unsafe)
        try:
            worker.requestInterruption()
        except Exception:
            pass
    
    def run_analysis(self):
        """Lancement de l'analyse optimisée"""
        if self.worker and self.worker.isRunning():
            return
        
        self.refresh_dataset_selector(keep_view=True)
        params = self.get_analysis_params()
        self._analysis_dataset = self.current_dataset
        
        # Afficher les optimisations actives
        if params['min_branch_length'] > 0.0:
            self.log(f"⚡ Minimum branch length: {params['min_branch_length']}px")
        if params['max_image_size'] > 0:
            self.log(f"⚡ Active resizing: max {params['max_image_size']}px")
        if params['connection_sample_rate'] < 0.1:
            self.log(f"⚡ Sampling: {params['connection_sample_rate']*100:.0f}%")
        
        self.worker = RootArchitectureWorker(self.mask_files, params)
        
        self.worker.progress.connect(self.update_progress)
        self.worker.day_analyzed.connect(self.store_day_data)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        
        # Désactiver les widgets pendant l'analyse
        widgets_to_disable = [
            getattr(self, "run_btn", None),
            getattr(self, "batch_btn", None),
            getattr(self, "select_btn", None),
            getattr(self, "dataset_selector", None),
            getattr(self, "dataset_selection_widget", None),
            getattr(self, "only_segmented_check", None)
        ]
        self._enter_busy(disable_widgets=[w for w in widgets_to_disable if w is not None])
        
        self._safe_set_enabled("run_btn", False)
        self._safe_set_enabled("select_btn", False)
        self._safe_set_enabled("stop_btn", True)
        self.progress_bar.setValue(0)
        # Reset only the data for the dataset being analyzed
        try:
            ds_key = self._analysis_dataset or self.current_dataset
            if not hasattr(self, "_results_cache") or self._results_cache is None:
                self._results_cache = {}
            self._results_cache.setdefault(ds_key, {"df": None, "daily_data": {}})["daily_data"] = {}
            # If the user is currently viewing the same dataset, clear the displayed data too
            if self.current_dataset == ds_key:
                self.daily_data = {}
                self.current_df = None
        except Exception:
            self.daily_data = {}
            self.current_df = None
        
        self.worker.start()
        self.log("Analysis started...")
    
    def _get_batch_dataset_names(self):
        """Retourne la liste de datasets à traiter en batch.
        
        Priorité:
        1) si un DatasetSelectionWidget est présent => datasets cochés
        2) sinon => tous les datasets qui ont des masques segmentés (roots)
        """
        file_extension = tuple([os.extsep + ext for ext in ['png', 'jpg', 'tif', 'tiff']])
        
        if not self.datasets:
            return []
        
        # 1) via widget de sélection (si présent)
        ds_widget = getattr(self, 'dataset_selector', None) or getattr(self, 'dataset_selection_widget', None)
        if ds_widget is not None:
            try:
                names = ds_widget.get_selected_dataset_names()
            except Exception:
                names = []
            names = [n for n in names if n in self.datasets]
            return sorted(names)
        
        # 2) fallback : tous les datasets avec masques roots
        names = []
        for ds_name, ds in self.datasets.items():
            try:
                seg_dir = ds.segmented_directory.get('roots', None)
            except Exception:
                continue
            if not seg_dir or not os.path.isdir(seg_dir):
                continue
            files = [f for f in os.listdir(seg_dir) if f.lower().endswith(file_extension)]
            if files:
                names.append(ds_name)
        return sorted(names)
    
    def batch_process_all_datasets(self):
        """Traite en batch uniquement les datasets cochés dans le widget."""
        batch_list = self._get_batch_dataset_names()
        if not batch_list:
            QMessageBox.information(
                self, "Batch",
                "No selected dataset (with segmented root masks) to process."
            )
            return
        
        # état batch
        self._batch_active = True
        self._batch_all_results = []  # DataFrames concaténés en fin de batch
        self._batch_queue = batch_list
        self._batch_index = 0
        self._batch_current = None
        
        self.log(f"🚀 Batch start: {len(self._batch_queue)} datasets")
        
        # désactiver boutons sensibles et dataset selector
        widgets_to_disable = [
            getattr(self, "run_btn", None),
            getattr(self, "batch_btn", None),
            getattr(self, "select_btn", None),
            getattr(self, "dataset_selector", None),
            getattr(self, "dataset_selection_widget", None),
            getattr(self, "only_segmented_check", None)
        ]
        
        # désactiver boutons sensibles
        self._enter_busy(disable_widgets=widgets_to_disable)
        self._safe_set_enabled("export_csv_btn", False)
        self._safe_set_enabled("export_images_btn", False)
        self._safe_set_enabled("batch_btn", False)
        self._stop_requested = False
        self._safe_set_enabled("stop_btn", True)
        
        self._batch_start_next()
    
    def _batch_start_next(self):
        if not getattr(self, "_batch_active", False):
            return
        
        if self._batch_index >= len(self._batch_queue):
            # fin batch
            self._batch_active = False
            self._safe_set_enabled("batch_btn", True)
            self._safe_set_enabled("run_btn", True)
            self.log("✅ Batch terminé")
            
            # s'assurer que la barre de progression arrive à 100%
            try:
                self.progress_bar.setValue(100)
            except Exception:
                pass
            
            # Export CSV global (tous datasets du batch)
            try:
                if getattr(self, "_batch_all_results", None):
                    batch_df = pd.concat(self._batch_all_results, ignore_index=True)
                    batch_csv = os.path.join(self.output_directory, "batch_root_architecture_results.csv")
                    batch_df.to_csv(batch_csv, sep=';', index=False)
                    self.log(f"💾 Batch CSV global: {batch_csv}")
            except Exception as e:
                self.log(f"⚠️ Impossible d'exporter le CSV global batch: {e}")
            
            QMessageBox.information(self, "Batch", "Batch processing terminé.")
            return
        
        ds_name = self._batch_queue[self._batch_index]
        self._batch_current = ds_name
        self._batch_index += 1
        
        # sélectionner le dataset (widget)
        if hasattr(self, 'dataset_selector'):
            self.dataset_selector.set_viewing_dataset(ds_name, emit_signal=True)
        else:
            # fallback: remplir mask_files directement
            seg_dir = self.datasets[ds_name].segmented_directory['roots']
            file_extension = tuple([os.extsep + ext for ext in ['png', 'jpg', 'tif', 'tiff']])
            self.mask_files = sorted([os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.lower().endswith(file_extension)])
            self.current_dataset = ds_name
        
        self.daily_data = {}
        self.current_df = None
        
        self.log(f"▶ Analyse: {ds_name} ({self._batch_index}/{len(self._batch_queue)})")
        
        # lance l’analyse normale (asynchrone)
        self.run_analysis()
    
    def _enter_busy(self, disable_widgets=None):
        """Enter a long-running 'busy' UI state (for async worker runs).
        
        We can't use a normal 'with ui_busy(...)' because the analysis runs in a QThread.
        So we manually enter/exit the context manager in run_analysis / analysis_finished / analysis_error.
        """
        # Close any previous context just in case
        self._exit_busy()
        try:
            self._busy_cm = ui_busy(self, disable_widgets=disable_widgets or [], set_wait_cursor=True)
            self._busy_cm.__enter__()
        except Exception:
            self._busy_cm = None
    
    def _exit_busy(self):
        """Leave busy UI state if active."""
        if self._busy_cm is None:
            return
        
        try:
            self._busy_cm.__exit__(None, None, None)
        except Exception:
            pass
        finally:
            self._busy_cm = None
            
            # Safety: fully restore any stacked override cursors (Qt keeps a stack)
            try:
                while QApplication.overrideCursor() is not None:
                    QApplication.restoreOverrideCursor()
            except Exception:
                pass
        
        # ------------------------------------------------------------
        # Batch orchestration (only if we are actually in batch mode)
        # ------------------------------------------------------------
        if not getattr(self, "_batch_running", False):
            return
        
        queue = getattr(self, "_batch_queue", None) or []
        idx = getattr(self, "_batch_index", 0)
        
        # If we're done (or queue empty), finish batch cleanly
        if idx >= len(queue):
            # Make sure UI/progress reflects completion
            try:
                if hasattr(self, "progress_bar"):
                    self.progress_bar.setValue(100)
            except Exception:
                pass
            
            self._batch_running = False
            self._batch_current = None
            
            # Optional: reset index so a new batch can start cleanly
            self._batch_index = len(queue)
            
            # Show completion popup (if you want it here)
            try:
                QMessageBox.information(self, "Batch terminé",
                                        f"Traitement batch terminé ({len(queue)} dataset(s)).")
            except Exception:
                pass
            
            self.log(f"✅ Batch terminé ({len(queue)} dataset(s))")
            return
        
        # Normal case: start next dataset
        ds_name = queue[idx]
        self._batch_current = ds_name
        self._batch_index = idx + 1
        
        # sélectionner le dataset (widget)
        if hasattr(self, 'dataset_selector'):
            self.dataset_selector.set_viewing_dataset(ds_name, emit_signal=True)
        else:
            # fallback: remplir mask_files directement
            seg_dir = self.datasets[ds_name].segmented_directory['roots']
            file_extension = tuple([os.extsep + ext for ext in ['png', 'jpg', 'tif', 'tiff']])
            self.mask_files = sorted([
                os.path.join(seg_dir, f)
                for f in os.listdir(seg_dir)
                if f.lower().endswith(file_extension)
            ])
            self.current_dataset = ds_name
        
        self.daily_data = {}
        self.current_df = None
        
        self.log(f"▶ Analyse: {ds_name} ({self._batch_index}/{len(queue)})")
        
        # lance l’analyse normale (asynchrone)
        self.run_analysis()
    
    def update_progress(self, value, message):
        self._safe_set_value("progress_bar", value)
        self._safe_set_text("status_label", message)
    
    def store_day_data(self, day, features, original, merged):
        ds_key = getattr(self, "_analysis_dataset", None) or self.current_dataset
        
        # Store into per-dataset cache
        try:
            if not hasattr(self, "_results_cache") or self._results_cache is None:
                self._results_cache = {}
            cache = self._results_cache.setdefault(ds_key, {"df": None, "daily_data": {}})
            cache_daily = cache.setdefault("daily_data", {})
            cache_daily[day] = (features, original, merged)
        except Exception:
            # Fallback (should not happen)
            self.daily_data[day] = (features, original, merged)
            cache_daily = self.daily_data
        
        self.log(f"✓ [{ds_key}] J{day}: {features.get('total_root_length', 0):.1f}px")
        
        # If the user is viewing a different dataset, do not overwrite the current visualization
        if self.current_dataset != ds_key:
            return
        
        # Point the 'current' view to this dataset's cached data
        self.daily_data = cache_daily
        
        # Slider range + initial value
        if len(self.daily_data) == 1:
            days = sorted(self.daily_data.keys())
            self.day_slider.setMinimum(min(days))
            self.day_slider.setMaximum(max(days))
            self.day_slider.setValue(min(days))
            
            # Keep heatmap day slider in sync (range + initial value)
            if hasattr(self, 'heatmap_day_slider') and self.heatmap_day_slider is not None:
                try:
                    self.heatmap_day_slider.blockSignals(True)
                    self.heatmap_day_slider.setMinimum(min(days))
                    self.heatmap_day_slider.setMaximum(max(days))
                    self.heatmap_day_slider.setValue(min(days))
                finally:
                    self.heatmap_day_slider.blockSignals(False)
                if hasattr(self, 'heatmap_day_label') and self.heatmap_day_label is not None:
                    self.heatmap_day_label.setText(str(min(days)))
        else:
            days = sorted(self.daily_data.keys())
            self.day_slider.setMaximum(max(days))
            if hasattr(self, 'heatmap_day_slider') and self.heatmap_day_slider is not None:
                try:
                    self.heatmap_day_slider.blockSignals(True)
                    self.heatmap_day_slider.setMaximum(max(days))
                finally:
                    self.heatmap_day_slider.blockSignals(False)
        
        # Refresh visuals for the currently viewed dataset/day
        self.update_day_visualization()
    
    def update_day_visualization(self):
        """Met à jour la visualisation pour le jour le plus proche ayant des données.
        
        Si le slider pointe sur un jour sans données, on "snap" automatiquement
        au jour disponible le plus proche, ce qui revient à rendre uniquement
        ces jours réellement sélectionnables.
        """
        if not self.daily_data:
            return
        
        requested_day = self.day_slider.value()
        available_days = sorted(self.daily_data.keys())
        
        # Trouver le jour le plus proche parmi ceux pour lesquels on a des données
        closest_day = min(available_days, key=lambda d: abs(d - requested_day))
        
        # Si le slider est sur un jour sans données, on le ramène sur le plus proche
        if closest_day != requested_day:
            self.day_slider.blockSignals(True)
            self.day_slider.setValue(closest_day)
            self.day_slider.blockSignals(False)
        
        day = closest_day
        self.day_label.setText(str(day))
        
        # Keep Heatmap day slider/label synchronized with the effective day.
        if hasattr(self, 'heatmap_day_slider') and self.heatmap_day_slider is not None:
            try:
                self.heatmap_day_slider.blockSignals(True)
                self.heatmap_day_slider.setMinimum(min(available_days))
                self.heatmap_day_slider.setMaximum(max(available_days))
                self.heatmap_day_slider.setValue(day)
            finally:
                self.heatmap_day_slider.blockSignals(False)
        if hasattr(self, 'heatmap_day_label') and self.heatmap_day_label is not None:
            self.heatmap_day_label.setText(str(day))
        
        features, original, merged = self.daily_data[day]
        self.viz_widget.set_day_data(day, features, original, merged, dataset=self.current_dataset)
        if hasattr(self, 'heatmap_widget') and self.heatmap_widget is not None:
            self.heatmap_widget.set_day_data(day, features, dataset=self.current_dataset)
    
    
    def _on_heatmap_day_changed(self, value):
        """When the Heatmap tab slider changes, drive the main day slider.
        
        The main day slider owns the 'snap to closest available day' logic via update_day_visualization().
        """
        if not self.daily_data:
            return
        if hasattr(self, 'day_slider') and self.day_slider is not None:
            try:
                self.day_slider.blockSignals(True)
                self.day_slider.setValue(int(value))
            finally:
                self.day_slider.blockSignals(False)
        # Apply snap + refresh
        self.update_day_visualization()
    
    
    def analysis_finished(self, df):
        # Ensure cursor/UI updates from worker signals settle
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        # Post-traitement du DataFrame : ajout d'une colonne de racines cumulatives
        if df is not None and not df.empty:
            sort_cols = [c for c in ['modality', 'day'] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols)
            
            subset_cols = [c for c in ['dataset','filename','day','modality'] if c in df.columns]
            if subset_cols:
                df = df.drop_duplicates(subset=subset_cols, keep='first')
            else:
                df = df.drop_duplicates(keep='first')
            
            def add_cummax(col_name: str, out_name: str):
                if col_name not in df.columns:
                    return
                if 'modality' in df.columns:
                    df[out_name] = df.groupby('modality')[col_name].cummax()
                else:
                    df[out_name] = df[col_name].cummax()
                df[out_name] = df[out_name].fillna(0).astype(int)
            
            add_cummax('root_count', 'root_count_cum')
            add_cummax('root_count_raw', 'root_count_raw_cum')
            add_cummax('root_count_attach', 'root_count_attach_cum')
        
        ds_key = getattr(self, "_analysis_dataset", None) or self.current_dataset
        # Store/refresh cache for the dataset that was analyzed
        try:
            if not hasattr(self, "_results_cache") or self._results_cache is None:
                self._results_cache = {}
            cache = self._results_cache.setdefault(ds_key, {"df": None, "daily_data": {}})
            cache["df"] = df
            # Ensure daily_data exists in cache even if no day signal was received (edge cases)
            cache.setdefault("daily_data", getattr(self, "daily_data", {}) or {})
        except Exception:
            pass
        
        # Only overwrite the currently displayed dataframe if the user is viewing the analyzed dataset
        if self.current_dataset == ds_key:
            self.current_df = df
        # Refresh UI after state changes (helpful if user is on other tabs)
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        # Make sure both day sliders have a valid range (helps when analysis ends while user stays on Heatmap tab)
        if getattr(self, 'daily_data', None):
            try:
                days = sorted(self.daily_data.keys())
                if days:
                    dmin, dmax = min(days), max(days)
                    self.day_slider.setMinimum(dmin)
                    self.day_slider.setMaximum(dmax)
                    if hasattr(self, 'heatmap_day_slider') and self.heatmap_day_slider is not None:
                        self.heatmap_day_slider.setMinimum(dmin)
                        self.heatmap_day_slider.setMaximum(dmax)
            except Exception:
                pass
        
        # ------------------------------------------------------------
        # Cache results per dataset (for instant refresh when switching datasets)
        # ------------------------------------------------------------
        try:
            if getattr(self, "current_dataset", None):
                # Shallow copies to avoid duplicating large numpy arrays
                _dd = {}
                try:
                    for _day, (_feat, _orig, _merged) in (self.daily_data or {}).items():
                        _dd[_day] = (dict(_feat) if isinstance(_feat, dict) else _feat, _orig, _merged)
                except Exception:
                    _dd = dict(self.daily_data) if isinstance(self.daily_data, dict) else {}
                _df = None
                try:
                    _df = self.current_df.copy() if self.current_df is not None else None
                except Exception:
                    _df = self.current_df
                if not hasattr(self, "_results_cache") or self._results_cache is None:
                    self._results_cache = {}
                self._results_cache[ds_key] = {"df": _df, "daily_data": _dd}
        except Exception:
            pass
        
        # ------------------------------------------------------------
        # Cache results per dataset (for instant refresh when switching datasets)
        # ------------------------------------------------------------
        try:
            if getattr(self, "current_dataset", None):
                # Shallow copies to avoid duplicating large numpy arrays
                _dd = {}
                try:
                    for _day, (_feat, _orig, _merged) in (self.daily_data or {}).items():
                        _dd[_day] = (dict(_feat) if isinstance(_feat, dict) else _feat, _orig, _merged)
                except Exception:
                    _dd = dict(self.daily_data) if isinstance(self.daily_data, dict) else {}
                _df = None
                try:
                    _df = self.current_df.copy() if self.current_df is not None else None
                except Exception:
                    _df = self.current_df
                if not hasattr(self, "_results_cache") or self._results_cache is None:
                    self._results_cache = {}
                self._results_cache[self.current_dataset] = {"df": _df, "daily_data": _dd}
        except Exception:
            pass
        
        # --------------------
        # Batch mode: auto-export + next dataset
        # --------------------
        if getattr(self, "_batch_active", False):
            ds = self.datasets[self.current_dataset]
            
            # dossier de sortie dataset (roots)
            out_dir = ds.result_directory.get('roots', None)
            if not out_dir:
                # fallback: crée un dossier Results/Roots dans le dataset
                out_dir = os.path.join(ds.path, "Results", "Roots")
                os.makedirs(out_dir, exist_ok=True)
                ds.result_directory['roots'] = out_dir
            
            # export CSV
            csv_path = os.path.join(out_dir, f"{self.current_dataset}_root_architecture_results.csv")
            self.current_df.to_csv(csv_path, sep=';', index=False)
            
            # export visualisations
            viz_dir = os.path.join(out_dir, "Visualisations")
            os.makedirs(viz_dir, exist_ok=True)
            self.export_visualizations_to(viz_dir)
            
            self.log(f"✅ Exported: {self.current_dataset} -> {out_dir}")
            # Ajouter au CSV global (batch)
            try:
                df_batch = self.current_df.copy()
                if 'dataset' not in df_batch.columns:
                    df_batch.insert(0, 'dataset', self.current_dataset)
                self._batch_all_results.append(df_batch)
            except Exception as e:
                self.log(f"⚠️ Batch aggregation failed: {e}")
            
            
            # rafraîchir l'UI avant de passer au dataset suivant
            self.populate_results_table(self.current_df)
            self.update_plot()
            self.update_day_visualization()
            QApplication.processEvents()
            
            # ✅ quitter busy maintenant que tout (DF + export + UI) est fait
            self._exit_busy()
            try:
                QApplication.processEvents()
            except Exception:
                pass
            
            # enchaîner le dataset suivant
            QTimer.singleShot(0, self._batch_start_next)
            return
        
        self._exit_busy()
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        self._analysis_dataset = None
        
        self._safe_set_enabled("export_csv_btn", True)
        self._safe_set_enabled("export_images_btn", True)
        
        self.populate_results_table(df)
        self.update_plot()
        self.update_day_visualization()
        
        self._safe_set_enabled("run_btn", True)
        self._safe_set_enabled("select_btn", True)
        self.progress_bar.setValue(100)
        self.status_label.setText("✓ Terminé")
        
        self.log(f"✅ Analyse terminée: {len(df)} échantillons")
        
        QMessageBox.information(
            self, "Analysis complete",
            f"✅ Analysis successfully completed!\n\n"
            f"📊 {len(df)} samples analyzed\n"
        )
    
    def analysis_error(self, error_msg):
        self._analysis_dataset = None
        self._safe_set_enabled("run_btn", True)
        self._safe_set_enabled("select_btn", True)
        self.log(f"❌ ERROR: {error_msg}")
        
        self._exit_busy()
        try:
            QApplication.processEvents()
        except Exception:
            pass
        
        # Remise de la barre de progression à 0%
        try:
            self.progress_bar.setValue(0)
        except Exception:
            pass
        
        QMessageBox.critical(
            self, "Analysis error:",
            f"An error has occurred:\n{error_msg}"
        )
    
    def populate_results_table(self, df):
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns.tolist())
        
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, float):
                    item = QTableWidgetItem(f"{value:.2f}")
                else:
                    item = QTableWidgetItem(str(value))
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
    
    def update_plot(self):
        if self.current_df is None:
            return
        
        variable = self.plot_combo.currentText()
        
        self.plot_figure.clear()
        ax = self.plot_figure.add_subplot(111)
        
        for modality in self.current_df['modality'].unique():
            subset = self.current_df[self.current_df['modality'] == modality]
            subset = subset.sort_values('day')
            
            ax.plot(subset['day'], subset[variable], 
                   marker='o', label=f'{modality}', linewidth=2)
        
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel(variable.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'Evolution - {variable.replace("_", " ").title()}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()
    
    def export_csv(self):
        if self.current_df is None:
            return
        
        default_path = os.path.join(self.output_directory, self.current_dataset + "-root_architecture_results.csv")
        filename, _ = QFileDialog.getSaveFileName(self, "Save results", default_path,
                                                  "CSV Files (*.csv)")
        
        if filename:
            self.current_df.to_csv(filename, index=False)
            self.log(f"💾 Exporté: {filename}")
            QMessageBox.information(self, "Export successful", f"Results saved:\n{filename}")
    
    # -------------------------
    # Export helpers
    # -------------------------
    def _export_target_pixels(self):
        """Renvoir (w_px, h_px, dpi) pour l'export des images de visualisation """
        # Default fallback
        sw, sh = 1600, 900
        try:
            screen = QApplication.primaryScreen()
            geom = screen.availableGeometry() if screen is not None else None
            if geom is not None:
                sw = int(geom.width())
                sh = int(geom.height())
        except Exception:
            pass
        
        # Target pixel size as a fraction of the screen (clamped)
        w_px = int(max(900, min(2200, sw * 0.92)))
        h_px = int(max(700, min(1650, sh * 0.88)))
        
        # Use a stable logical dpi (do NOT use physical DPI, it shrinks figures on HiDPI)
        dpi = 120
        
        return w_px, h_px, dpi
    
    def _make_export_figure(self):
        """Create a Matplotlib Figure sized in *pixels* (via figsize+dpi)."""
        w_px, h_px, dpi = self._export_target_pixels()
        try:
            fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, constrained_layout=True)
        except TypeError:
            fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        return fig, dpi
    
    
    def export_visualizations(self):
        if not self.daily_data:
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory",
        )
        
        if not output_dir:
            return
        
        self.export_visualizations_to(output_dir)
        
        QMessageBox.information(
            self, "Export successful",
            f"{len(self.daily_data)} exported visualizations"
        )
    
    def export_visualizations_to(self, output_dir: str):
        """Export des visualisations (2x2) sans boîte de dialogue (mode batch)."""
        if not self.daily_data:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for day in sorted(self.daily_data.keys()):
            features, original, merged = self.daily_data[day]
            
            fig = Figure(figsize=(15, 10))
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            
            ax1.imshow(original, cmap='gray')
            ax1.set_title(f'Original - J{day}')
            ax1.axis('off')
            
            ax2.imshow(merged, cmap='gray')
            ax2.set_title('Merged')
            ax2.axis('off')
            
            if len(features.get('skeleton', [])) > 0:
                ax3.imshow(features['skeleton'], cmap='gray')
                
                if len(features.get('convex_hull', [])) > 0:
                    ax3.imshow(features['convex_hull'], cmap='Reds', alpha=0.2)
                
                if len(features.get('main_path', [])) > 0:
                    main_path = features['main_path']
                    ax3.plot(main_path[:, 1], main_path[:, 0], 'r-', linewidth=2)
                
                if len(features.get('endpoints', [])) > 0:
                    endpoints = features['endpoints']
                    ax3.scatter(endpoints[:, 1], endpoints[:, 0], c='yellow', s=30, 
                              marker='o', edgecolors='red', linewidths=1)
                
                ax3.set_title('Skeleton')
                ax3.axis('off')
            
            ax4.axis('off')
            dpi_save = 300
            try:
                fig_w_px = float(fig.get_figwidth()) * float(dpi_save)
            except Exception:
                fig_w_px = 0.0
            
            # Consider the final saved image width (figsize * dpi_save)
            compact_view = (fig_w_px > 0 and fig_w_px < 3300)
            
            if fig_w_px >= 5400:
                stats_fs = 10
            elif fig_w_px >= 4200:
                stats_fs = 9
            else:
                stats_fs = 7 if compact_view else 8
            
            stats_text = (
                f"Day {day}\n"
                f"Dataset: {self.current_dataset}\n\n"
                f"Main root length: {features['main_root_length']:.1f} px\n"
                f"Secondary roots length: {features['secondary_roots_length']:.1f} px\n"
                f"Total length (graph): {features['total_root_length']:.1f} px\n"
                f"Exact skeleton length: {features.get('exact_skeleton_length', 0):.1f} px\n\n"
                f"Number of branches: {features['branch_count']}\n"
                f"Number of end points without pruning: {features['endpoint_count_raw']}\n"
                f"Number of roots without pruning: {features['root_count_raw']}\n"
                f"Number of end points: {features['endpoint_count']}\n"
                f"Number of roots: {features['root_count']}\n"
                f"Root count attach: {features['root_count_attach']}\n\n"
                f"Mean secondary angles: {features['mean_secondary_angles']:.2f}\n"
                f"Standard deviation secondary angles: {features['std_secondary_angles']:.2f}\n"
                f"Mean absolute secondary angles: {features['mean_abs_secondary_angles']:.2f}\n"
                f"Std absolute secondary angles: {features['std_abs_secondary_angles']:.2f}\n\n"
                f"Total area: {features['total_area']:.0f} px²\n"
                f"Convex area: {features['convex_area']:.0f} px²\n\n"
                f"Barycenter: ({features['centroid_x']:.1f}, {features['centroid_y']:.1f})"
            )
            
            # --- Grid: add min/mean/max summary (no heatmap) ---
            gl = features.get('grid_lengths', None)
            if isinstance(gl, np.ndarray) and gl.ndim == 2 and gl.size > 0:
                try:
                    gmin = float(np.min(gl))
                    gmean = float(np.mean(gl))
                    gmax = float(np.max(gl))
                    r, c = gl.shape[0], gl.shape[1]
                    
                    if compact_view:
                        stats_text += (
                            f"\n\nGrid ({r}x{c}): min/mean/max = {gmin:.2f}/{gmean:.2f}/{gmax:.2f} px"
                        )
                    else:
                        stats_text += (
                            f"\n\nGrid per cell ({r}x{c}):\n"
                            f"  - Min: {gmin:.2f} px\n"
                            f"  - Mean: {gmean:.2f} px\n"
                            f"  - Max: {gmax:.2f} px"
                        )
                except Exception:
                    pass
            
            # Background box that fills the whole stats axis (better alignment than default bbox)
            try:
                ax4.add_patch(
                    FancyBboxPatch(
                        (0.0, 0.0), 1.0, 1.0,
                        boxstyle='round,pad=0.02',
                        transform=ax4.transAxes,
                        linewidth=0,
                        facecolor='wheat',
                        alpha=0.5,
                        zorder=0,
                    )
                )
            except Exception:
                pass
            
            ax4.text(
                0.03, 0.97,
                stats_text,
                transform=ax4.transAxes,
                fontsize=stats_fs,
                verticalalignment='top',
                horizontalalignment='left',
                fontfamily='monospace',
                wrap=True,
                zorder=1,
            )
            fig.tight_layout()
            
            save_path = output_path / f"{self.current_dataset}_root_arch_J{day:03d}.png"
            fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        self.log(f"💾 {len(self.daily_data)} exported images (batch)")
     
    def log(self, message):
        """Log robuste: fonctionne même si l'UI n'est pas encore construite."""
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(message)
            try:
                self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
            except Exception:
                pass
        else:
            print(message)
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = RootArchitectureWindow()
    window.show()
    
    if PYQT_VERSION == 6:
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())
        # Grille (pour longueurs par section)
        try:
            params["grid_rows"] = int(self.viz_widget.n_rows_grid_combo.currentText())
            params["grid_cols"] = int(self.viz_widget.n_cols_grid_combo.currentText())
        except Exception:
            params["grid_rows"] = None
            params["grid_cols"] = None