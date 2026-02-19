import cv2, os, sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
import pandas as pd
import seaborn as sns
import tifffile as tiff

# Configuration du style seaborn
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)

from collections import deque
from datetime import datetime, timedelta
from skimage.morphology import convex_hull_image, remove_small_objects, skeletonize

PYQT_AVAILABLE = False
PYQT_VERSION = None

try:
    from PyQt6.QtCore import Qt, QPoint, QRect, QTimer, QSize, QLocale, pyqtSignal, QThread, QObject, pyqtSlot
    from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QToolButton, QPushButton, QMessageBox, QLineEdit, QFileDialog, QInputDialog, QTextEdit, QGridLayout, QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QMenu, QMenuBar, QSizePolicy, QProgressDialog, QTabWidget, QListWidget, QListWidgetItem, QSplitter, QDialog, QProgressBar, QScrollArea
    from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QBitmap, QPainter, QRegion, QImage, QIntValidator, QDoubleValidator, QKeySequence, QAction
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
except:
    try:
        from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QSize, QLocale, pyqtSignal, QThread, QObject, pyqtSlot
        from PyQt5.QtWidgets import QApplication, QWidget, QAction, QMainWindow, QLabel, QToolButton, QPushButton, QMessageBox, QLineEdit, QFileDialog, QInputDialog, QTextEdit, QGridLayout, QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QMenu, QMenuBar, QSizePolicy, QProgressDialog, QTabWidget, QListWidget, QListWidgetItem, QSplitter, QDialog, QProgressBar, QScrollArea
        from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QBitmap, QPainter, QRegion, QImage, QIntValidator, QDoubleValidator, QKeySequence
        PYQT_AVAILABLE = True
        PYQT_VERSION = 5
    except:
        print("This script requires PyQt5 or PyQt6 to run. Neither of these versions was found!")

from utils import *
from widgets import *
from window_analyzer import *


# Export des données relatives à l'aire des enveloppes convexes et du nombre de pixels représentant l'objet segmenté (racines ou feuilles) dans un fichier CSV
def export_convex_area(output_path, data_dict, csv_separator=';'):
    header = list(data_dict.keys())
    n_images = len(data_dict[header[0]])
    with open(output_path, 'w') as csv_file:
        csv_file.write(csv_separator.join(header) + "\n")
        for idx_row in range(n_images):
            current_row = [str(data_dict[col_name][idx_row]) for col_name in header]
            csv_file.write(csv_separator.join(current_row) + "\n")
    return(None)


# Read tif and other image format
def image_read(image_path):
    tif_extension = tuple([os.extsep + ext for ext in ['tif', 'tiff']])
    if image_path.endswith(tif_extension):
        image = tiff.imread(image_path)
    else:
        image = cv2.imread(image_path)
    
    return image


# Replace file extension
def replace_file_extension(file_name, new_extension):
    return (os.extsep).join(file_name.split(os.extsep)[:-1]) + os.extsep + new_extension


def ensure_rgb(image, path_hint=None):
    """ S'assure qu'une image est bien retournée au format RGB (H,W,3). On suppose :
    - cv2.imread() retourne une image au format BGR pour les images non TIF
    - tifffile.imread() retourne une image au format RGB
    On convertit donc les images BGR en RGB pour les images non TIF.
    """
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        return image
    if image.ndim == 2:
        # grayscale -> 3-channel RGB for consistent display in threshold windows
        return np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 3:
        if path_hint:
            p = str(path_hint).lower()
            if p.endswith(('.tif', '.tiff')):
                return image
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            return image
    return image


class ProcessingWorker(QThread):
    """Thread worker pour le traitement en arrière-plan avec signaux de progression"""
    progress_update = pyqtSignal(str, int, int)  # message, current, total
    step_completed = pyqtSignal(str)  # message de fin d'étape
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent_app, selected_datasets, active_analyses, crop=True, segment=True):
        super().__init__()
        self.parent_app = parent_app
        self.crop = crop
        self.segment = segment
        self.selected_datasets = selected_datasets
        self.active_analyses = active_analyses
        self.should_stop = False
        self.kernel_shape_dict = {"Rectangle":cv2.MORPH_RECT, "Cross":cv2.MORPH_CROSS, "Ellipse":cv2.MORPH_ELLIPSE}
        self.kernel_keys_list = list(self.kernel_shape_dict.keys())
    
    def run(self):
        try:
            total_steps = len(self.selected_datasets) * len(self.active_analyses) * (int(self.crop) + int(self.segment))  # crop + segment
            
            current_step = 0
            
            for analysis_type in self.active_analyses:
                if self.should_stop:
                    break
                
                # Phase 1: Cropping (seulement si demandé)
                if self.crop:
                    self.progress_update.emit(f"Cropping {analysis_type} images...", current_step, total_steps)
                    
                    config = self.parent_app.analysis_configs[analysis_type]
                    if not config.selection_coords and self.parent_app.current_image_array is not None:
                        height, width = self.parent_app.current_image_array.shape[:2]
                        config.selection_coords = QRect(0, 0, width, height)
                    
                    for dataset_name in self.selected_datasets:
                        if self.should_stop:
                            break
                        
                        dataset_info = self.parent_app.datasets[dataset_name]
                        self.progress_update.emit(f"Cropping {analysis_type} for {dataset_name}...", current_step, total_steps)
                        
                        # Charger les images de ce dataset
                        dataset_images = [f for f in os.listdir(dataset_info.path) 
                                        if f.lower().endswith(self.parent_app.image_extension_list)]
                        
                        for i, image_name in enumerate(dataset_images):
                            if self.should_stop:
                                break
                            
                            image_path = os.path.join(dataset_info.path, image_name)
                            image = image_read(image_path)
                            
                            rect = config.selection_coords
                            crop_image = image[rect.y():rect.y() + rect.height(),
                                             rect.x():rect.x() + rect.width()]
                            
                            output_name = replace_file_extension(image_name, new_extension='png')
                            output_path = os.path.join(dataset_info.crop_directory[analysis_type], output_name)
                            cv2.imwrite(output_path, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
                            
                            # Mise à jour fine de la progression
                            sub_progress = int((i + 1) / len(dataset_images) * 100)
                            self.progress_update.emit(f"Cropping {analysis_type} for {dataset_name}: {i+1}/{len(dataset_images)}", 
                                                    current_step, total_steps)
                        
                        current_step += 1
                    
                    self.step_completed.emit(f"Cropping {analysis_type} completed")
                
                # Phase 2: Segmentation (seulement si demandée)
                if self.segment:    
                    self.progress_update.emit(f"Segmenting {analysis_type} images...", current_step, total_steps)
                    
                    # Stocker le type pour l'export
                    self.parent_app.current_threshold_type = analysis_type
                    
                    if analysis_type not in self.parent_app.analysis_dict:
                        self.parent_app.analysis_dict[analysis_type] = {}
                    
                    for dataset_name in self.selected_datasets:
                        if self.should_stop:
                            break
                        
                        dataset_info = self.parent_app.datasets[dataset_name]
                        self.progress_update.emit(f"Segmenting {analysis_type} for {dataset_name}...", current_step, total_steps)
                        
                        self._segment_dataset(dataset_name, dataset_info, analysis_type)
                        if analysis_type == 'roots':
                            self._skeletonize_dataset(dataset_name, dataset_info, analysis_type)
                        
                        current_step += 1
                    
                    self.step_completed.emit(f"Segmentation {analysis_type} completed")
                    
                    # Créer le CSV global
                    self.parent_app.createGlobalCSV(analysis_type)
                    self.step_completed.emit(f"Global CSV for {analysis_type} created")
            
            self.finished_processing.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _segment_dataset(self, dataset_name, dataset_info, analysis_type):
        """Segmente un dataset spécifique"""
        config = self.parent_app.analysis_configs[analysis_type]
        params = config.threshold_params
        
        # Déterminer quelles images utiliser
        if (analysis_type in dataset_info.crop_directory and 
            dataset_info.crop_directory[analysis_type] and 
            os.path.exists(dataset_info.crop_directory[analysis_type])):
            source_directory = dataset_info.crop_directory[analysis_type]
        else:
            source_directory = dataset_info.path
        
        data_dict = {
            "Dataset": [], "Image name": [], "Analysis type": [],
            "Pixel count": [], "Convex area": [], "Modality": [], "Day": []
        }
        
        # Traiter chaque image
        image_list = [f for f in os.listdir(source_directory) 
                     if f.lower().endswith(self.parent_app.image_extension_list)]
        
        old_mask = None
        for i, image_name in enumerate(image_list):
            if self.should_stop:
                break
            
            image_path = os.path.join(source_directory, image_name)
            
            # Extraction des informations du nom
            split_name = image_name.split(os.extsep)[0].split('_')
            image_moda, image_day = 'NaN', 'NaN'
            for idx, name_part in enumerate(split_name):
                if idx == 0:
                    image_moda = name_part
                else:
                    if name_part[0].lower() == 'j':
                        image_day = int(name_part[1:])
            
            # Charger et traiter l'image
            image = cv2.cvtColor(image_read(image_path), cv2.COLOR_BGR2RGB)
            
            # Appliquer le seuillage RGB
            binary_image = self.parent_app.applyColorThreshold(image, params)
            
            # Closing operation and keep ax connected components
            if params['kernel_size'] != (0, 0):
                kernel_shape_name = self.kernel_keys_list[params['kernel_shape']]
                kernel = cv2.getStructuringElement(self.kernel_shape_dict[kernel_shape_name], params['kernel_size'])
                binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            
            if params['keep_max_component']:
                retval, labels = cv2.connectedComponents(binary_image)
                
                label_count = np.zeros(retval, dtype='int32')
                for k in range(1, retval):
                    label_count[k] = np.sum(labels == k)
                
                max_label = np.argmax(label_count)
                binary_image = np.array(labels == max_label, dtype='uint8') * 255
            
            if params['min_connected_components_area'] > 0:
                (n_labels, label_ids, values, centroid)  = cv2.connectedComponentsWithStats(binary_image, 8)
                clean_mask = np.zeros_like(binary_image, dtype='uint8')
                for j in range(1, n_labels):
                    area = values[j, cv2.CC_STAT_AREA]
                    if area > params['min_connected_components_area']:
                        component_mask = (label_ids == j).astype("uint8") * 255
                        clean_mask = cv2.bitwise_or(clean_mask, component_mask)
                
                clean_mask = np.array(clean_mask > 0, dtype='uint8') * 255
                
                if params['max_centroid_dst'] > 0.0:
                    diff_mask = binary_image - clean_mask
                    (n_labels_diff, label_ids_diff, values_diff, centroid_diff)  = cv2.connectedComponentsWithStats(diff_mask, 8)
                    (n_labels_clean, label_ids_clean, values_clean, centroid_clean)  = cv2.connectedComponentsWithStats(clean_mask, 8)
                    
                    dst_centroid = np.zeros([n_labels_diff-1, n_labels_clean-1], dtype='float32')
                    for k in range(1, n_labels_diff):
                        for l in range(1, n_labels_clean):
                            dst_centroid[k-1, l-1] = lg.norm(np.abs(centroid_diff[k, :] - centroid_clean[l, :]))
                    
                    min_dst = np.min(dst_centroid, axis=1)
                    
                    idx_keep_comp = np.where(min_dst < params['max_centroid_dst'])[0]
                    new_mask = np.zeros_like(image, dtype='uint8')
                    for idx_comp in idx_keep_comp:
                        comp_mask = (label_ids_diff == idx_comp).astype("uint8") * 255
                        new_mask = cv2.bitwise_or(new_mask, comp_mask)
                    
                    clean_mask = cv2.bitwise_or(new_mask, clean_mask)
                
                binary_image = cv2.bitwise_and(binary_image, clean_mask)
            
            # Suppression du bruit
            if params['min_object_size'] > 0:
                binary_image = remove_small_objects(
                    np.array(binary_image > 0, dtype='bool'), 
                    min_size=params['min_object_size']
                )
                binary_image = (binary_image * 255).astype('uint8')
            
            # On garde les pixels de l'ancien masque
            if i > 0:
                binary_image = (np.logical_or(binary_image, old_mask) * 255).astype('uint8')
            
            old_mask = binary_image.copy()
            # Calcul de l'enveloppe convexe
            hull_mask = (convex_hull_image(binary_image > 0) * 255).astype(np.uint8)
            convex_area = np.sum(hull_mask == 255)
            pixel_count = np.sum(binary_image > 127)
            
            # Sauvegarder les données
            data_dict["Dataset"].append(dataset_name)
            data_dict["Analysis type"].append(analysis_type)
            data_dict["Modality"].append(image_moda)
            data_dict["Day"].append(image_day)
            data_dict["Image name"].append(image_name)
            data_dict["Pixel count"].append(pixel_count)
            data_dict["Convex area"].append(convex_area)
            
            # Sauvegarder l'image segmentée
            output_name = replace_file_extension(image_name, new_extension='png')
            output_path = os.path.join(dataset_info.segmented_directory[analysis_type], output_name)
            cv2.imwrite(output_path, binary_image)
            
            # Sauvegarder l'enveloppe convexe
            output_path = os.path.join(dataset_info.convex_hull_directory[analysis_type], output_name)
            cv2.imwrite(output_path, hull_mask)
            
            # Mise à jour de la progression
            self.progress_update.emit(f"Segmenting {dataset_name}: {i+1}/{len(image_list)}", 0, 1)
        
        # Envoie des données à la classe 'parent'
        self.parent_app.analysis_dict[analysis_type][dataset_name] = data_dict
        # Sauvegarder les données CSV pour ce dataset
        csv_filename = f"{dataset_name}_{analysis_type}_analysis.csv"
        output_csv_path = os.path.join(os.path.dirname(dataset_info.segmented_directory[analysis_type]), csv_filename)
        export_convex_area(output_csv_path, data_dict, csv_separator=';')
    
    def _skeletonize_dataset(self, dataset_name, dataset_info, analysis_type):
        if (analysis_type in dataset_info.segmented_directory and 
            dataset_info.segmented_directory[analysis_type] and 
            os.path.exists(dataset_info.segmented_directory[analysis_type])):
            source_directory = dataset_info.segmented_directory[analysis_type]
        else:
            source_directory = dataset_info.path
        
        # Traiter chaque image
        image_list = [f for f in os.listdir(source_directory) 
                     if f.lower().endswith(self.parent_app.image_extension_list)]
        
        for i, image_name in enumerate(image_list):
            if self.should_stop:
                break
            
            image_path = os.path.join(source_directory, image_name)
            
            # Charger et traiter l'image
            image = image_read(image_path)
            if image.ndim > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            skeleton_image = np.array(skeletonize(image), dtype='uint8') * 255
            
            output_name = replace_file_extension(image_name, new_extension='png')
            output_path = os.path.join(dataset_info.skeletonized_directory[analysis_type], output_name)
            cv2.imwrite(output_path, skeleton_image)
    
    def stop(self):
        self.should_stop = True




class ProgressWindow(QDialog):
    """Fenêtre de progression détaillée pour le traitement"""
    
    def __init__(self, parent=None, crop_only=False, segment_only=False):
        super().__init__(parent)
        self.crop_only = crop_only
        self.segment_only = segment_only
        self.crop_datasets = crop_only or (not self.segment_only)
        self.segment_datasets = (not self.crop_only) or segment_only
        self.setWindowTitle("Processing Progress")
        self.setModal(True)
        self.setMinimumSize(500, 200)
        
        layout = QVBoxLayout(self)
        
        # Titre
        self.title_label = QLabel("Processing datasets...")
        self.title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Message de statut
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Barre de progression principale
        self.main_progress = QProgressBar()
        self.main_progress.setTextVisible(True)
        layout.addWidget(self.main_progress)
        
        # Zone de log
        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(150)
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)
        
        # Bouton d'annulation
        button_layout = QHBoxLayout()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_processing)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.worker = None
    
    def start_processing(self, parent_app, selected_datasets, active_analyses):
        """Démarre le traitement avec le worker thread"""
        self.worker = ProcessingWorker(parent_app, selected_datasets, active_analyses, crop=self.crop_datasets, segment=self.segment_datasets)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.step_completed.connect(self.add_log_message)
        self.worker.finished_processing.connect(self.processing_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
    
    @pyqtSlot(str, int, int)
    def update_progress(self, message, current, total):
        """Met à jour la barre de progression"""
        self.status_label.setText(message)
        if total > 0:
            self.main_progress.setMaximum(total)
            self.main_progress.setValue(current)
        else:
            self.main_progress.setRange(0, 0)  # Mode indéterminé
    
    @pyqtSlot(str)
    def add_log_message(self, message):
        """Ajoute un message au log"""
        self.log_area.append(f"✓ {message}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    
    @pyqtSlot()
    def processing_finished(self):
        """Appelé quand le traitement est terminé"""
        self.status_label.setText("Processing completed successfully!")
        self.main_progress.setValue(self.main_progress.maximum())
        self.add_log_message("All processing completed successfully!")
        self.cancel_button.setText("Close")
    
    @pyqtSlot(str)
    def handle_error(self, error_message):
        """Gère les erreurs du traitement"""
        self.status_label.setText("Error occurred during processing")
        self.add_log_message(f"Error: {error_message}")
        self.cancel_button.setText("Close")
    
    def cancel_processing(self):
        """Annule le traitement en cours"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()  # Attendre que le thread se termine
            self.add_log_message("Processing cancelled by user")
        self.accept()


class AnalysisConfig:
    """Configuration pour un type d'analyse (racines ou feuilles)"""
    def __init__(self, name, check_state=False, init_threshold_params=None):
        self.name = name
        self.enabled = check_state
        self.selection_coords = None
        self.default_threshold_params = {
                                            'red_range': (0, 255),
                                            'green_range': (0, 255), 
                                            'blue_range': (0, 255),
                                            'red_invert': False,
                                            'green_invert': False,
                                            'blue_invert': False,
                                            'min_branch_length': 20.0,
                                            'max_connection_dst': 240,
                                            'keep_max_component': False,
                                            'min_connected_components_area': 0,
                                            'max_centroid_dst': 0.0,
                                            'min_object_size': 200,
                                            'kernel_size': 5,
                                            'kernel_shape': cv2.MORPH_RECT,
                                            'fusion_masks': True
                                        }
        
        self.threshold_params = {}
        if init_threshold_params is None:
            self.threshold_params = self.default_threshold_params
        else:
            for key_name, key_value in self.default_threshold_params.items():
                self.threshold_params[key_name] = init_threshold_params.get(key_name, key_value)

class DatasetInfo:
    """Informations sur un jeu de données"""
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.crop_directory = {}
        self.segmented_directory = {}
        self.skeletonized_directory = {}
        self.convex_hull_directory = {}
        self.result_directory = {}
        self.selected = True  # Par défaut, tous les datasets sont sélectionnés


class ImageDisplayWidget(QLabel):
    """Widget personnalisé pour l'affichage d'images avec gestion correcte des coordonnées"""
    
    def __init__(self, parent=None, display_selection=True, default_color=[255, 0, 0]):
        super().__init__(parent)
        self.parent_app = parent
        self.display_selection = display_selection
        self.setMinimumSize(400, 300)
        self.setScaledContents(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding if PYQT_VERSION == 6 else QSizePolicy.Expanding, QSizePolicy.Policy.Expanding if PYQT_VERSION == 6 else QSizePolicy.Expanding)
        self.setStyleSheet("border: 1px solid gray;")
        
        # Variables pour la sélection
        self.selection_active = False
        self.start_point = None
        self.current_point = None
        self.selection_rect = None
        self.original_selection_coords = None
        self.default_color = default_color
        self.rgb_paint_color = QColor(self.default_color[0], self.default_color[1], self.default_color[2], 255)
        
        # Variables pour l'image
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.image_array = None
        self.image_rect = QRect()
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
    def setImagePixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.updateDisplayedImage()
    
    def setImageArray(self, image_array):
        """Définit l'image à partir d'un numpy array et stocke les données"""
        if image_array is None:
            return
        
        self.image_array = image_array.copy()  # Stocker une copie de l'array
        
        # Convertir numpy array vers QPixmap
        height, width = image_array.shape[:2]
        if len(image_array.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888 if PYQT_VERSION == 6 else QImage.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8 if PYQT_VERSION == 6 else QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        self.setImagePixmap(pixmap)
    
    def updateDisplayedImage(self):
        if self.original_pixmap is None:
            return
            
        widget_size = self.size()
        pixmap_size = self.original_pixmap.size()
        
        scale_x = widget_size.width() / pixmap_size.width()
        scale_y = widget_size.height() / pixmap_size.height()
        self.scale_factor = min(scale_x, scale_y)
        
        scaled_width = int(pixmap_size.width() * self.scale_factor)
        scaled_height = int(pixmap_size.height() * self.scale_factor)
        
        self.offset_x = (widget_size.width() - scaled_width) // 2
        self.offset_y = (widget_size.height() - scaled_height) // 2
        
        self.scaled_pixmap = self.original_pixmap.scaled(
            scaled_width, scaled_height, 
            Qt.AspectRatioMode.KeepAspectRatio if PYQT_VERSION == 6 else Qt.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation if PYQT_VERSION == 6 else Qt.SmoothTransformation
        )
        
        self.image_rect = QRect(self.offset_x, self.offset_y, scaled_width, scaled_height)
        self.updateSelectionAfterResize()
        self.update()
    
    def updateSelectionAfterResize(self):
        if self.original_selection_coords is not None:
            self.selection_rect = self.convertOriginalToDisplayCoords(self.original_selection_coords)
    
    def convertOriginalToDisplayCoords(self, original_rect):
        if original_rect is None or self.scale_factor == 0:
            return None
        
        display_rect = QRect(
            int(original_rect.x() * self.scale_factor + self.offset_x),
            int(original_rect.y() * self.scale_factor + self.offset_y),
            int(original_rect.width() * self.scale_factor),
            int(original_rect.height() * self.scale_factor)
        )
        return display_rect
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateDisplayedImage()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        if self.scaled_pixmap is not None:
            painter.drawPixmap(self.offset_x, self.offset_y, self.scaled_pixmap)
        
        if self.display_selection and (self.selection_rect is not None):
            painter.setPen(self.rgb_paint_color)
            painter.setBrush(Qt.BrushStyle.NoBrush if PYQT_VERSION == 6 else Qt.NoBrush)
            painter.drawRect(self.selection_rect)
    
    def changeColorSelection(self, new_rgb_color):
        if isinstance(new_rgb_color, list):
            try:
                if len(new_rgb_color) < 4:
                    new_rgb_color += [255]*(4-len(new_rgb_color))
                self.rgb_paint_color = QColor(new_rgb_color[0], new_rgb_color[1], new_rgb_color[2], new_rgb_color[3])
            except:
                print("Invalid RGB color : a list of at least 3 integers is expected!")
    
    def mousePressEvent(self, event):
        if not self.parent_app or not self.parent_app.activeSelection:
            return
            
        if event.button() == (Qt.MouseButton.LeftButton if PYQT_VERSION == 6 else Qt.LeftButton) and self.isPointInImage(event.pos()):
            self.selection_active = True
            self.start_point = event.pos()
            self.current_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.current_point).normalized()
            self.update()
    
    def mouseMoveEvent(self, event):
        if self.selection_active and self.start_point is not None:
            constrained_pos = self.constrainPointToImage(event.pos())
            self.current_point = constrained_pos
            self.selection_rect = QRect(self.start_point, self.current_point).normalized()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if self.selection_active and event.button() == (Qt.MouseButton.LeftButton if PYQT_VERSION == 6 else Qt.LeftButton):
            self.selection_active = False
            if self.selection_rect is not None:
                real_coords = self.getOriginalImageCoordinates(self.selection_rect)
                self.original_selection_coords = real_coords
                if self.parent_app:
                    self.parent_app.onSelectionFinished(real_coords)
    
    def isPointInImage(self, point):
        return self.image_rect.contains(point)
    
    def constrainPointToImage(self, point):
        x = max(self.image_rect.left(), min(point.x(), self.image_rect.right()))
        y = max(self.image_rect.top(), min(point.y(), self.image_rect.bottom()))
        return QPoint(x, y)
    
    def getOriginalImageCoordinates(self, selection_rect):
        if self.original_pixmap is None or self.scale_factor == 0:
            return None
        
        adjusted_rect = QRect(
            selection_rect.x() - self.offset_x,
            selection_rect.y() - self.offset_y,
            selection_rect.width(),
            selection_rect.height()
        )
        
        original_rect = QRect(
            int(adjusted_rect.x() / self.scale_factor),
            int(adjusted_rect.y() / self.scale_factor),
            int(adjusted_rect.width() / self.scale_factor),
            int(adjusted_rect.height() / self.scale_factor)
        )
        return original_rect
    
    def setSelectionCoords(self, coords):
        """Définit les coordonnées de sélection depuis l'extérieur"""
        self.original_selection_coords = coords
        if coords:
            self.selection_rect = self.convertOriginalToDisplayCoords(coords)
        else:
            self.selection_rect = None
        self.update()
    
    def clearImage(self):
        """Efface l'image affichée"""
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.image_array = None
        self.update()
    
    def clearSelection(self):
        self.selection_rect = None
        self.selection_active = False
        self.start_point = None
        self.current_point = None
        self.original_selection_coords = None
        self.update()
    
    def getDisplayedImage(self):
        """Retourne l'image actuellement affichée dans le widget sous forme de numpy array"""
        return self.image_array



class App(QWidget):
    def __init__(self, current_script_name, current_directory, screen_size=None, parent=None):
        super(App, self).__init__(parent)
        self.screen_size = screen_size
        self.parent = parent
        self.app_title = "Roots System Analyzer"
        self.current_script_name = current_script_name
        self.current_directory = current_directory
        self.root_arch_window = None
        
        # Variables d'interface
        self.activeSelection = False
        self.current_analysis_type = "roots"  # "roots" ou "leaves"
        self.rgb_color = [56, 166, 239]
        self.current_tab_index = 0
        
        # Variable pour stocker la sélection temporaire en cours
        self.temp_selection_coords = None
        
        # Configuration des analyses (globale pour tous les datasets)
        self.roots_checkbox_analysis_state = True
        self.leaves_checkbox_analysis_state = True
        self.analysis_configs = {
            "roots": AnalysisConfig("Roots", self.roots_checkbox_analysis_state, {
                'red_range': (0, 180),
                'green_range': (0, 255),
                'blue_range': (0, 255),
                'red_invert': True,
                'green_invert': False,
                'blue_invert': False,
                'min_branch_length': 10.0,
                'keep_max_component': False,
                'min_connected_components_area': 0,
                'max_centroid_dst': 0,
                'min_object_size': 0,
                'kernel_size': 0,
                'kernel_shape': cv2.MORPH_RECT,
                'fusion_masks' : False
            }),
            "leaves": AnalysisConfig("Leaves", self.leaves_checkbox_analysis_state, {
                'red_range': (0, 255),
                'green_range': (100, 255),
                'blue_range': (0, 255),
                'red_invert': False,
                'green_invert': False,
                'blue_invert': False,
                'min_branch_length': 10.0,
                'keep_max_component': False,
                'min_connected_components_area': 0,
                'max_centroid_dst': 0,
                'min_object_size': 0,
                'kernel_size': 0,
                'kernel_shape': cv2.MORPH_RECT,
                'fusion_masks' : False
            })
        }
        self.cleaning_mask_parameters = { 'closing_radius': 5,
                                          'closing_shape': cv2.MORPH_RECT,
                                          'min_branch_length': 10.0,
                                          'keep_max_component': False,
                                          'min_connected_components_area': 0,
                                          'max_centroid_dst': 0,
                                          'min_object_size': 0,
                                          'line_thickness': 5,
                                          'temporal_merge': True,
                                          'connect_objects': True
                                        }
        
        # Gestion des datasets multiples
        self.datasets = {}  # Dict[str, DatasetInfo]
        self.current_dataset = None
        self.base_input_directory = None
        self.base_output_directory = None
        self.analysis_dict = {}
        
        # Répertoires
        self.icons_directory = os.path.join(self.current_directory, "icons")
        self.default_image_directory = os.path.join(self.current_directory, "Data")
        if not os.path.exists(self.default_image_directory):
            self.default_image_directory = self.current_directory
        
        self.default_output_directory = os.path.join(self.current_directory, "Analysis")
        os.makedirs(self.default_output_directory, exist_ok=True)
        
        # Logo
        self.logo_max_width = 280
        if PYQT_VERSION == 6:
            self.logo_file_name = "logo_rtt_2.png"
            self.logo_file_path = os.path.join(self.icons_directory, self.logo_file_name)
            if os.path.exists(self.logo_file_path):
                self.logo_pixmap = QPixmap(self.logo_file_path)
            else:
                self.logo_file_name = "logo_rtt.png"
                self.logo_file_path = os.path.join(self.icons_directory, self.logo_file_name)
                self.logo_pixmap = QPixmap(self.logo_file_path)
        else:
            self.logo_file_name = "logo_rtt.png"
            self.logo_file_path = os.path.join(self.icons_directory, self.logo_file_name)
            self.logo_pixmap = QPixmap(self.logo_file_path)
        
        self.logo_start_file_name = "logo_start.png"
        self.logo_start_file_path = os.path.join(self.icons_directory, self.logo_start_file_name)
        self.logo_start_shape = cv2.imread(self.logo_start_file_path, cv2.IMREAD_UNCHANGED).shape[:2]
        self.start_icon_qsize = QSize(round(self.logo_start_shape[0] * 200 / self.logo_start_shape[1]), 200)
        self.logo_start_icon = QIcon(self.logo_start_file_path)
        
        # Images du dataset courant
        self.image_name_list = []
        self.image_path_list = []
        self.image_extension_list = ('bmp', 'png', 'jpg', 'tif', 'tiff')
        self.n_images_loaded = 0
        self.idx_current_image = 0
        
        # Image courante
        self.current_image_name = None
        self.current_image_array = None
        self.current_image_mode = "original"  # "original", "roots_cropped", "leaves_cropped"
        self.displayed_image_arrays = {}  # Cache des images affichées
        
        # Interface
        self._createInterface()
    
    def _createInterface(self):
        main_layout = QHBoxLayout(self)
        
        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        main_layout.addWidget(splitter)
        # Panel de gauche (dans un QScrollArea pour les petites résolutions)
        left_panel = self._createLeftPanel()
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame if PYQT_VERSION == 6 else QFrame.NoFrame)
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
        
        # Empêche le panneau de gauche d'être écrasé verticalement : on scroll au lieu de compresser
        left_scroll.setMinimumWidth(320)
        splitter.addWidget(left_scroll)
        
        # Panel de droite (image avec onglets)
        right_panel = self._createRightPanel()
        splitter.addWidget(right_panel)
        
        # Proportions
        splitter.setSizes([300, 1200])
    
    def _createLeftPanel(self):
        panel = QFrame()
        panel.setMinimumWidth(300)
        panel.setFrameStyle(QFrame.Shape.StyledPanel if PYQT_VERSION == 6 else QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Titre
        title = QLabel("Multi-Dataset Analysis Tool")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(title)
        
        # Boutons de dossier
        file_layout = QVBoxLayout()
        self.importBaseDirectoryButton = QPushButton("Select datasets folder")
        self.importBaseDirectoryButton.clicked.connect(self.openBaseDirectoryDialog)
        file_layout.addWidget(self.importBaseDirectoryButton)
        
        self.exportBaseDirectoryButton = QPushButton("Select output folder")
        self.exportBaseDirectoryButton.clicked.connect(self.openOutputDirectoryDialog)
        file_layout.addWidget(self.exportBaseDirectoryButton)
        
        layout.addLayout(file_layout)
        
        # Explorateur de datasets avec sélection
        datasets_layout = QVBoxLayout()
        datasets_title = QLabel("Datasets:")
        datasets_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        datasets_layout.addWidget(datasets_title)
        
        # Boutons de sélection globale
        selection_buttons_layout = QHBoxLayout()
        self.selectAllButton = QPushButton("Select All")
        self.selectAllButton.clicked.connect(self.selectAllDatasets)
        self.deselectAllButton = QPushButton("Deselect All")
        self.deselectAllButton.clicked.connect(self.deselectAllDatasets)
        selection_buttons_layout.addWidget(self.selectAllButton)
        selection_buttons_layout.addWidget(self.deselectAllButton)
        datasets_layout.addLayout(selection_buttons_layout)
        
        self.datasetsList = QListWidget()
        self.datasetsList.setMaximumHeight(150)
        self.datasetsList.itemClicked.connect(self.onDatasetClicked)
        self.datasetsList.itemChanged.connect(self.onDatasetSelectionChanged)
        datasets_layout.addWidget(self.datasetsList)
        
        self.currentDatasetLabel = QLabel("No dataset selected")
        self.currentDatasetLabel.setStyleSheet("font-style: italic; color: gray;")
        datasets_layout.addWidget(self.currentDatasetLabel)
        
        # Compteur de datasets sélectionnés
        self.selectedCountLabel = QLabel("Selected: 0/0")
        self.selectedCountLabel.setStyleSheet("font-size: 10px; color: blue;")
        datasets_layout.addWidget(self.selectedCountLabel)
        
        layout.addLayout(datasets_layout)
        
        # Séparateur
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine if PYQT_VERSION == 6 else QFrame.HLine)
        layout.addWidget(separator1)
        
        # Sélection du type d'analyse
        analysis_layout = QVBoxLayout()
        analysis_title = QLabel("Analysis Type:")
        analysis_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        analysis_layout.addWidget(analysis_title)
        
        self.rootsRadio = QCheckBox("Roots Analysis")
        self.rootsRadio.setChecked(True)
        self.rootsRadio.stateChanged.connect(lambda: self.toggleAnalysisType("roots"))
        analysis_layout.addWidget(self.rootsRadio)
        
        self.leavesRadio = QCheckBox("Leaves Analysis")
        self.leavesRadio.setChecked(True)
        self.leavesRadio.stateChanged.connect(lambda: self.toggleAnalysisType("leaves"))
        analysis_layout.addWidget(self.leavesRadio)
        
        layout.addLayout(analysis_layout)
        
        # Séparateur
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine if PYQT_VERSION == 6 else QFrame.HLine)
        layout.addWidget(separator2)
        
        # Outils de sélection
        tools_layout = QVBoxLayout()
        tools_title = QLabel("Selection Tools:")
        tools_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        tools_layout.addWidget(tools_title)
        
        tool_buttons_layout = QHBoxLayout()
        self.colorChoiceButton = QPushButton()
        self.colorChoiceButton.setFixedSize(32, 32)
        self.colorChoiceButton.setStyleSheet(f"background-color: rgb({self.rgb_color[0]}, {self.rgb_color[1]}, {self.rgb_color[2]})")
        self.colorChoiceButton.clicked.connect(self.openColorDialog)
        tool_buttons_layout.addWidget(self.colorChoiceButton)
        
        self.imageSelectionToolButton = QPushButton("Selection")
        self.imageSelectionToolButton.setCheckable(True)
        self.imageSelectionToolButton.clicked.connect(self.activateRectangleSelection)
        tool_buttons_layout.addWidget(self.imageSelectionToolButton)
        
        tools_layout.addLayout(tool_buttons_layout)
        layout.addLayout(tools_layout)
        
        # Informations de sélection
        self.selectionInfoLabel = QLabel("No selection")
        self.selectionInfoLabel.setWordWrap(True)
        self.selectionInfoLabel.setStyleSheet("border: 1px solid gray; padding: 5px;")
        layout.addWidget(self.selectionInfoLabel)
        
        # NOUVEAUX BOUTONS DE CONFIRMATION DE SÉLECTION
        confirm_layout = QVBoxLayout()
        confirm_title = QLabel("Confirm Selections:")
        confirm_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        confirm_layout.addWidget(confirm_title)
        
        confirm_buttons_layout = QHBoxLayout()
        self.confirmRootsSelectionButton = QPushButton("Confirm Roots")
        self.confirmRootsSelectionButton.clicked.connect(lambda: self.confirmSelection("roots"))
        self.confirmRootsSelectionButton.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.confirmRootsSelectionButton.setEnabled(False)
        confirm_buttons_layout.addWidget(self.confirmRootsSelectionButton)
        
        self.confirmLeavesSelectionButton = QPushButton("Confirm Leaves")
        self.confirmLeavesSelectionButton.clicked.connect(lambda: self.confirmSelection("leaves"))
        self.confirmLeavesSelectionButton.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.confirmLeavesSelectionButton.setEnabled(False)
        confirm_buttons_layout.addWidget(self.confirmLeavesSelectionButton)
        
        confirm_layout.addLayout(confirm_buttons_layout)
        layout.addLayout(confirm_layout)
        
        # Séparateur
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine if PYQT_VERSION == 6 else QFrame.HLine)
        layout.addWidget(separator3)
        
        # Actions par type d'analyse (MODIFIÉES)
        actions_layout = QVBoxLayout()
        actions_title = QLabel("Actions:")
        actions_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        actions_layout.addWidget(actions_title)
        
        # Actions pour racines (crop seulement)
        roots_actions = QHBoxLayout()
        self.cropRootsButton = QPushButton("Crop Roots")
        self.cropRootsButton.clicked.connect(lambda: self.cropAllDatasets("roots"))
        self.segmentRootsButton = QPushButton("Segment Roots")
        self.segmentRootsButton.clicked.connect(lambda: self.openThresholdWindow("roots"))
        roots_actions.addWidget(self.cropRootsButton)
        roots_actions.addWidget(self.segmentRootsButton)
        actions_layout.addLayout(roots_actions)
        
        # Actions pour feuilles (crop seulement)
        leaves_actions = QHBoxLayout()
        self.cropLeavesButton = QPushButton("Crop Leaves")
        self.cropLeavesButton.clicked.connect(lambda: self.cropAllDatasets("leaves"))
        self.segmentLeavesButton = QPushButton("Segment Leaves")
        self.segmentLeavesButton.clicked.connect(lambda: self.openThresholdWindow("leaves"))
        leaves_actions.addWidget(self.cropLeavesButton)
        leaves_actions.addWidget(self.segmentLeavesButton)
        actions_layout.addLayout(leaves_actions)
        
        # Roots tracking actions
        roots_tracking_actions = QVBoxLayout()        
        self.startRootsArchitectureAnalysisButton = QPushButton("Roots architecture analysis")
        self.startRootsArchitectureAnalysisButton.clicked.connect(self.openRootArchitecture)
        self.startRootsArchitectureAnalysisButton.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        roots_tracking_actions.addWidget(self.startRootsArchitectureAnalysisButton)
        
        actions_layout.addLayout(roots_tracking_actions)
        layout.addLayout(actions_layout)
        
        # Séparateur
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine if PYQT_VERSION == 6 else QFrame.HLine)
        layout.addWidget(separator3)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.prevButton = QPushButton("Previous")
        self.prevButton.clicked.connect(self.previousImage)
        nav_layout.addWidget(self.prevButton)
        
        self.nextButton = QPushButton("Next")
        self.nextButton.clicked.connect(self.nextImage)
        nav_layout.addWidget(self.nextButton)
        
        layout.addLayout(nav_layout)
        
        # Index d'image
        self.imageIndexLabel = QLabel("Image: 0 / 0")
        layout.addWidget(self.imageIndexLabel)
        
        # Logo
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.Shape.HLine if PYQT_VERSION == 6 else QFrame.HLine)
        layout.addWidget(separator4)
        
        # --- Logo centré et responsive ---
        self.logo_label = QLabel()
        self.logo_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter
        )
        
        # Important : ne pas fixer directement scaledToWidth ici
        self.logo_label.setPixmap(self.logo_pixmap)
        
        layout.addStretch()
        layout.addWidget(self.logo_label)
        
        # Forcer une mise à jour au resize
        panel.installEventFilter(self)
        
        return panel
    
    def _createRightPanel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Titre
        self.imageNameTitle = QLabel("Image name")
        self.imageNameTitle.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        self.imageNameTitle.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(self.imageNameTitle)
        
        # Onglets pour les différentes vues
        self.tabWidget = QTabWidget()
        
        # Onglet image originale
        self.originalTab = ImageDisplayWidget(self, display_selection=True, default_color=self.rgb_color)
        self.tabWidget.addTab(self.originalTab, "Original")
        
        # Onglet racines
        self.rootsTab = ImageDisplayWidget(self, display_selection=False, default_color=self.rgb_color)
        self.tabWidget.addTab(self.rootsTab, "Roots")
        
        # Onglet feuilles
        self.leavesTab = ImageDisplayWidget(self, display_selection=False, default_color=self.rgb_color)
        self.tabWidget.addTab(self.leavesTab, "Leaves")
        
        # Onglet graphiques racines
        self.rootsGraphTab = QWidget()
        self.tabWidget.addTab(self.rootsGraphTab, "Roots Graph")
        self.roots_figure = plt.figure(figsize=(12, 8))
        self.roots_canvas = FigureCanvas(self.roots_figure)
        roots_graph_layout = QVBoxLayout()
        roots_graph_layout.addWidget(self.roots_canvas)
        self.rootsGraphTab.setLayout(roots_graph_layout)
        
        # Onglet graphiques feuilles
        self.leavesGraphTab = QWidget()
        self.tabWidget.addTab(self.leavesGraphTab, "Leaves Graph")
        self.leaves_figure = plt.figure(figsize=(12, 8))
        self.leaves_canvas = FigureCanvas(self.leaves_figure)
        leaves_graph_layout = QVBoxLayout()
        leaves_graph_layout.addWidget(self.leaves_canvas)
        self.leavesGraphTab.setLayout(leaves_graph_layout)
        
        self.tabWidget.currentChanged.connect(self.onTabChanged)
        
        layout.addWidget(self.tabWidget)
        
        return widget
    
    def eventFilter(self, obj, event):
        if event.type() == event.Type.Resize:
            self.updateLogoSize()
        return super().eventFilter(obj, event)
    
    def updateLogoSize(self):
        """Redimensionne le logo pour qu’il reste centré et adaptatif"""
        if not hasattr(self, "logo_label") or self.logo_pixmap is None:
            return
        
        # Largeur disponible dans le panneau gauche
        available_width = self.logo_label.parent().width()
        
        # Taille finale = min(largeur panneau, max)
        target_width = min(self.logo_max_width, available_width - 30)
        
        scaled = self.logo_pixmap.scaledToWidth(
            target_width,
            Qt.TransformationMode.SmoothTransformation
            if PYQT_VERSION == 6 else Qt.SmoothTransformation
        )
        
        self.logo_label.setPixmap(scaled)
    
    # Confirmation de sélection
    def confirmSelection(self, analysis_type):
        """Confirme et sauvegarde la sélection pour un type d'analyse"""
        # Si aucun sélection, on garde l'entièreté de l'image
        if self.temp_selection_coords is None:
            nr, nc = self.current_image_array.shape[:2]
            self.temp_selection_coords = QRect(0, 0, nc, nr)
        
        if self.temp_selection_coords is not None:
            self.analysis_configs[analysis_type].selection_coords = self.temp_selection_coords
            self.current_analysis_type = analysis_type
            # Mettre à jour l'affichage
            self.updateAllSelections()
            
            # Message de confirmation
            QMessageBox.information(self, "Selection Confirmed", 
                                  f"{analysis_type.title()} selection confirmed!\n"
                                  f"X: {self.temp_selection_coords.x()}, Y: {self.temp_selection_coords.y()}\n"
                                  f"W: {self.temp_selection_coords.width()}, H: {self.temp_selection_coords.height()}")
            
            print(f"{analysis_type.title()} selection confirmed: {self.analysis_configs[analysis_type].selection_coords}")
            
            # Désactiver le bouton jusqu'à la prochaine sélection
            if analysis_type == "roots":
                self.confirmRootsSelectionButton.setEnabled(False)
            else:
                self.confirmLeavesSelectionButton.setEnabled(False)
        else:
            QMessageBox.warning(self, "No Selection", 
                              f"Please make a selection first before confirming {analysis_type} area.")
    
    def openBaseDirectoryDialog(self):
        """Sélectionne le dossier principal contenant tous les datasets"""
        directory = QFileDialog.getExistingDirectory(self, "Select datasets folder", self.default_image_directory)
        if directory:
            self.base_input_directory = os.path.normpath(directory)
            self.scanForDatasets()
            self.updateDatasetsList()
    
    def openOutputDirectoryDialog(self):
        """Sélectionne le dossier de sortie principal"""
        directory = QFileDialog.getExistingDirectory(self, "Select output directory", self.default_output_directory)
        if directory:
            self.base_output_directory = os.path.normpath(directory)
            self.createAllOutputDirectories()
    
    def scanForDatasets(self):
        """Scanne le dossier principal pour trouver tous les datasets"""
        if not self.base_input_directory or not os.path.exists(self.base_input_directory):
            return
        
        self.datasets.clear()
        
        # Chercher tous les sous-dossiers contenant des images
        for item in os.listdir(self.base_input_directory):
            item_path = os.path.join(self.base_input_directory, item)
            if os.path.isdir(item_path):
                # Vérifier s'il y a des images dans ce dossier
                images = [f for f in os.listdir(item_path) 
                         if f.lower().endswith(self.image_extension_list)]
                if images:
                    self.datasets[item] = DatasetInfo(item, item_path)
        
        print(f"Found {len(self.datasets)} datasets: {list(self.datasets.keys())}")
    
    def updateDatasetsList(self):
        """Met à jour la liste des datasets dans l'interface avec checkboxes"""
        self.datasetsList.clear()
        for dataset_name in sorted(self.datasets.keys()):
            item = QListWidgetItem(dataset_name)
            item.setFlags(item.flags() | (Qt.ItemFlag.ItemIsUserCheckable if PYQT_VERSION == 6 else Qt.ItemIsUserCheckable))
            item.setCheckState((Qt.CheckState.Checked if PYQT_VERSION == 6 else Qt.Checked) if self.datasets[dataset_name].selected else (Qt.CheckState.Unchecked if PYQT_VERSION == 6 else Qt.Unchecked))
            self.datasetsList.addItem(item)
        
        # Sélectionner le premier dataset par défaut pour l'affichage
        if self.datasets and self.datasetsList.count() > 0:
            self.datasetsList.setCurrentRow(0)
            first_dataset_name = list(self.datasets.keys())[0]
            self.selectDatasetForViewing(first_dataset_name)
        
        self.updateSelectedCount()
    
    def onDatasetClicked(self, item):
        """Appelé quand un dataset est cliqué dans la liste (pour l'affichage)"""
        dataset_name = item.text()
        self.selectDatasetForViewing(dataset_name)
        
        # Rafraîchir le graphe si on est sur un onglet graphe
        current_tab_index = self.tabWidget.currentIndex()
        if current_tab_index == 3:  # Roots Graph tab
            self.createSpecificGraph("roots")
        elif current_tab_index == 4:  # Leaves Graph tab
            self.createSpecificGraph("leaves")   
    
    def onDatasetSelectionChanged(self, item):
        """Appelé quand la checkbox d'un dataset change"""
        dataset_name = item.text()
        if dataset_name in self.datasets:
            self.datasets[dataset_name].selected = (item.checkState() == (Qt.CheckState.Checked if PYQT_VERSION == 6 else Qt.Checked))
        self.updateSelectedCount()
    
    def selectAllDatasets(self):
        """Sélectionne tous les datasets"""
        for i in range(self.datasetsList.count()):
            item = self.datasetsList.item(i)
            item.setCheckState(Qt.CheckState.Checked if PYQT_VERSION == 6 else Qt.Checked)
            dataset_name = item.text()
            if dataset_name in self.datasets:
                self.datasets[dataset_name].selected = True
        self.updateSelectedCount()
    
    def deselectAllDatasets(self):
        """Désélectionne tous les datasets"""
        for i in range(self.datasetsList.count()):
            item = self.datasetsList.item(i)
            item.setCheckState(Qt.CheckState.Unchecked if PYQT_VERSION == 6 else Qt.Unchecked)
            dataset_name = item.text()
            if dataset_name in self.datasets:
                self.datasets[dataset_name].selected = False
        self.updateSelectedCount()
    
    def updateSelectedCount(self):
        """Met à jour le compteur de datasets sélectionnés"""
        selected_count = sum(1 for dataset in self.datasets.values() if dataset.selected)
        total_count = len(self.datasets)
        self.selectedCountLabel.setText(f"Selected: {selected_count}/{total_count}")
    
    def selectDatasetForViewing(self, dataset_name):
        """Sélectionne un dataset pour l'affichage (sans changer sa sélection pour le traitement)"""
        if dataset_name not in self.datasets:
            return
        
        self.current_dataset = self.datasets[dataset_name]
        self.currentDatasetLabel.setText(f"Viewing: {dataset_name}")
        
        # Charger les images de ce dataset
        self.loadDatasetImages()
        
        # Afficher la première image
        if self.n_images_loaded > 0:
            self.idx_current_image = 0
            self.displayImageIndex(0)
            self.updateImageIndexLabel()
    
    def getSelectedDatasets(self):
        """Retourne la liste des noms des datasets sélectionnés"""
        return [name for name, dataset in self.datasets.items() if dataset.selected]
    
    def loadDatasetImages(self):
        """Charge la liste des images du dataset courant"""
        if not self.current_dataset:
            if self.confirmRootsSelectionButton.isEnabled():
                self.confirmRootsSelectionButton.setEnabled(False)
            if self.confirmLeavesSelectionButton.isEnabled():
                self.confirmLeavesSelectionButton.setEnabled(False)
            return
        
        dataset_path = self.current_dataset.path
        self.image_name_list = [f for f in os.listdir(dataset_path) 
                              if f.lower().endswith(self.image_extension_list)]
        self.image_path_list = [os.path.join(dataset_path, name) 
                              for name in self.image_name_list]
        self.n_images_loaded = len(self.image_name_list)
        
        if self.n_images_loaded > 0:
            if not self.confirmRootsSelectionButton.isEnabled():
                self.confirmRootsSelectionButton.setEnabled(True)
            if not self.confirmLeavesSelectionButton.isEnabled():
                self.confirmLeavesSelectionButton.setEnabled(True)
        
        print(f"Loaded {self.n_images_loaded} images from dataset '{self.current_dataset.name}'")
    
    def createAllOutputDirectories(self):
        """Crée tous les répertoires de sortie pour tous les datasets"""
        if not (self.base_output_directory and self.datasets):
            return
        
        for dataset_name, dataset_info in self.datasets.items():
            base_output = os.path.join(self.base_output_directory, dataset_name)
            
            # Créer les dossiers pour racines
            roots_dir = os.path.join(base_output, "Roots")
            dataset_info.crop_directory["roots"] = os.path.join(roots_dir, "Crop")
            dataset_info.segmented_directory["roots"] = os.path.join(roots_dir, "Segmented")
            dataset_info.skeletonized_directory["roots"] = os.path.join(roots_dir, "Skeletonized")
            dataset_info.convex_hull_directory["roots"] = os.path.join(roots_dir, "ConvexHull")
            dataset_info.result_directory["roots"] = os.path.join(roots_dir, "Results")
            
            # Créer les dossiers pour feuilles
            leaves_dir = os.path.join(base_output, "Leaves")
            dataset_info.crop_directory["leaves"] = os.path.join(leaves_dir, "Crop")
            dataset_info.segmented_directory["leaves"] = os.path.join(leaves_dir, "Segmented")
            dataset_info.convex_hull_directory["leaves"] = os.path.join(leaves_dir, "ConvexHull")
            dataset_info.result_directory["leaves"] = os.path.join(leaves_dir, "Results")
            
            # Créer les répertoires
            os.makedirs(dataset_info.crop_directory["roots"], exist_ok=True)
            os.makedirs(dataset_info.segmented_directory["roots"], exist_ok=True)
            os.makedirs(dataset_info.skeletonized_directory["roots"], exist_ok=True)
            os.makedirs(dataset_info.convex_hull_directory["roots"], exist_ok=True)
            os.makedirs(dataset_info.result_directory["roots"], exist_ok=True)
            
            os.makedirs(dataset_info.crop_directory["leaves"], exist_ok=True)
            os.makedirs(dataset_info.segmented_directory["leaves"], exist_ok=True)
            os.makedirs(dataset_info.convex_hull_directory["leaves"], exist_ok=True)
            os.makedirs(dataset_info.result_directory["leaves"], exist_ok=True)
        
        print("Output directories created for all datasets")
    
    def toggleAnalysisType(self, analysis_type):
        """Active/désactive un type d'analyse"""
        checkbox = self.rootsRadio if analysis_type == "roots" else self.leavesRadio
        self.analysis_configs[analysis_type].enabled = checkbox.isChecked()
        
        # Activer les boutons correspondants
        if analysis_type == "roots":
            self.cropRootsButton.setEnabled(checkbox.isChecked())
            self.segmentRootsButton.setEnabled(checkbox.isChecked())
        else:
            self.cropLeavesButton.setEnabled(checkbox.isChecked())
            self.segmentLeavesButton.setEnabled(checkbox.isChecked())
        
        # Mettre à jour le type d'analyse courant
        if checkbox.isChecked():
            self.current_analysis_type = analysis_type
    
    def onTabChanged(self, index):
        """Appelé quand l'onglet change"""
        tab_names = ["original", "roots", "leaves", "roots graph", "leaves graph"]
        self.current_image_mode = tab_names[index]
        self.current_tab_index = index
        
        # Mettre à jour la sélection affichée selon l'onglet ou le graphe si l'onglet sélectionné est un onglet de graphe
        self.activeSelection = self.imageSelectionToolButton.isChecked()
        
        current_widget = self.tabWidget.currentWidget()
        if index == 1:  # Roots tab
            current_widget.setSelectionCoords(self.analysis_configs["roots"].selection_coords)
        elif index == 2:  # Leaves tab
            current_widget.setSelectionCoords(self.analysis_configs["leaves"].selection_coords)
        elif index == 3:  # Roots Graph tab
            self.createSpecificGraph("roots")
        elif index == 4:  # Leaves Graph tab
            self.createSpecificGraph("leaves")
        else:  # Original tab
            # Afficher la sélection du type d'analyse actif
            if not self.activeSelection:
                current_widget.clearSelection()
            elif self.current_analysis_type in self.analysis_configs:
                current_widget.setSelectionCoords(self.analysis_configs[self.current_analysis_type].selection_coords)
    
    def createSpecificGraph(self, analysis_type):
        """Crée un graphique spécifique avec seaborn pour un rendu moderne"""
        if not self.current_dataset or not hasattr(self, 'analysis_dict'):
            return
        
        # Sélectionner la figure appropriée
        if analysis_type == "roots":
            figure = self.roots_figure
            canvas = self.roots_canvas
            color_palette = "viridis"  # Palette verte pour les racines
        else:
            figure = self.leaves_figure
            canvas = self.leaves_canvas
            color_palette = "plasma"   # Palette chaude pour les feuilles
        
        # Effacer le graphique précédent
        figure.clear()
        
        dataset_name = self.current_dataset.name
        
        # Vérifier si on a des données d'analyse pour ce type
        if (analysis_type in self.analysis_dict and 
            dataset_name in self.analysis_dict[analysis_type]):
            
            current_res = self.analysis_dict[analysis_type][dataset_name]
            
            # Créer un DataFrame pandas à partir des données
            df = pd.DataFrame(current_res)
            
            # Nettoyer et convertir les données
            df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
            df["Pixel count"] = pd.to_numeric(df["Pixel count"], errors="coerce")
            df["Convex area"] = pd.to_numeric(df["Convex area"], errors="coerce")
            
            # Supprimer les lignes avec des valeurs manquantes
            df = df.dropna(subset=["Day", "Pixel count", "Convex area"])
            
            if len(df) > 0:
                # Créer deux subplots avec seaborn
                ax1 = figure.add_subplot(2, 1, 1)
                ax2 = figure.add_subplot(2, 1, 2)
                
                # Définir la palette de couleurs
                n_modalities = df['Modality'].nunique()
                colors = sns.color_palette(color_palette, n_modalities)
                
                # Graphique 1: Pixel Count avec seaborn
                sns.lineplot(data=df, 
                            x='Day', 
                            y='Pixel count', 
                            hue='Modality',
                            marker='o',
                            markersize=8,
                            linewidth=3,
                            palette=colors,
                            ax=ax1)
                
                # Ajouter des barres d'erreur manuellement
                for i, modality in enumerate(df['Modality'].unique()):
                    subset = df[df['Modality'] == modality].groupby('Day')['Pixel count'].agg(['mean', 'std', 'count']).reset_index()
                    subset['se'] = subset['std'] / np.sqrt(subset['count'])
                    subset['se'] = subset['se'].fillna(0)
                    
                    ax1.errorbar(subset['Day'], subset['mean'], 
                                yerr=subset['se'],
                                fmt='none',
                                capsize=5,
                                capthick=2,
                                color=colors[i],
                                alpha=0.7)
                
                # Graphique 2: Convex Area avec seaborn
                sns.lineplot(data=df, 
                            x='Day', 
                            y='Convex area', 
                            hue='Modality',
                            marker='s',
                            markersize=8,
                            linewidth=3,
                            palette=colors,
                            ax=ax2)
                
                # Ajouter des barres d'erreur
                for i, modality in enumerate(df['Modality'].unique()):
                    subset = df[df['Modality'] == modality].groupby('Day')['Convex area'].agg(['mean', 'std', 'count']).reset_index()
                    subset['se'] = subset['std'] / np.sqrt(subset['count'])
                    subset['se'] = subset['se'].fillna(0)
                    
                    ax2.errorbar(subset['Day'], subset['mean'], 
                                yerr=subset['se'],
                                fmt='none',
                                capsize=5,
                                capthick=2,
                                color=colors[i],
                                alpha=0.7)
                
                # Configuration des graphiques avec style moderne
                ax1.set_title(f'{analysis_type.title()} Analysis - Pixel Count Evolution\nDataset: {dataset_name}', 
                             fontsize=14, fontweight='bold', pad=20)
                ax1.set_xlabel('Day', fontsize=12, fontweight='semibold')
                ax1.set_ylabel('Pixel Count', fontsize=12, fontweight='semibold')
                ax1.legend(title='Modality', title_fontsize=11, fontsize=10, frameon=True, fancybox=True, shadow=True)
                
                ax2.set_title(f'{analysis_type.title()} Analysis - Convex Area Evolution', 
                             fontsize=14, fontweight='bold', pad=20)
                ax2.set_xlabel('Day', fontsize=12, fontweight='semibold')
                ax2.set_ylabel('Convex Area', fontsize=12, fontweight='semibold')
                ax2.legend(title='Modality', title_fontsize=11, fontsize=10, frameon=True, fancybox=True, shadow=True)
                
                # Améliorer l'apparence avec seaborn
                sns.despine(ax=ax1, top=True, right=True)
                sns.despine(ax=ax2, top=True, right=True)
                
                # Ajouter une grille subtile
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax2.grid(True, alpha=0.3, linestyle='--')
                
                # Optionnel: ajouter une zone d'ombre pour montrer la tendance
                if len(df['Day'].unique()) > 2:  # Seulement s'il y a assez de points
                    try:
                        for i, modality in enumerate(df['Modality'].unique()):
                            mod_data = df[df['Modality'] == modality].sort_values('Day')
                            if len(mod_data) > 2:
                                # Régression polynomiale légère pour la tendance
                                z1 = np.polyfit(mod_data['Day'], mod_data['Pixel count'], min(2, len(mod_data)-1))
                                p1 = np.poly1d(z1)
                                x_smooth = np.linspace(mod_data['Day'].min(), mod_data['Day'].max(), 100)
                                ax1.plot(x_smooth, p1(x_smooth), '--', alpha=0.5, color=colors[i], linewidth=2)
                                
                                z2 = np.polyfit(mod_data['Day'], mod_data['Convex area'], min(2, len(mod_data)-1))
                                p2 = np.poly1d(z2)
                                ax2.plot(x_smooth, p2(x_smooth), '--', alpha=0.5, color=colors[i], linewidth=2)
                    except:
                        pass  # Si la régression échoue, on continue sans
                
            else:
                # Pas de données valides
                ax = figure.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f'No valid {analysis_type} data available\n\nRun {analysis_type} analysis first to generate graphs', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=16, 
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                ax.set_title(f"{analysis_type.title()} Analysis - Dataset: {dataset_name}", 
                           fontsize=14, fontweight='bold')
                ax.axis('off')
        else:
            # Pas de données d'analyse
            ax = figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'No {analysis_type} analysis data available\n\nRun {analysis_type} analysis first to see beautiful graphs here!', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=16,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax.set_title(f"{analysis_type.title()} Analysis - Dataset: {dataset_name}", 
                       fontsize=14, fontweight='bold')
            ax.axis('off')
        
        # Ajuster la mise en page et redessiner
        figure.tight_layout(pad=3.0)
        canvas.draw()
    
    def displayImageIndex(self, index):
        """Affiche l'image à l'index donné du dataset courant"""
        if not self.current_dataset or not (0 <= index < self.n_images_loaded):
            return
        
        image_path = self.image_path_list[index]
        self.current_image_name = self.image_name_list[index]
        
        # Charger l'image originale
        self.current_image_array = image_read(image_path)
        
        # Convertir en QPixmap
        height, width, channel = self.current_image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888 if PYQT_VERSION == 6 else QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Afficher dans l'onglet original
        self.originalTab.setImagePixmap(pixmap)
        self.originalTab.setImageArray(self.current_image_array)
        
        # Charger les images croppées si elles existent
        self.loadCroppedImages()
        
        self.imageNameTitle.setText(f"{self.current_dataset.name} - {os.path.splitext(self.current_image_name)[0]}")
        self.updateAllSelections()
    
    def loadCroppedImages(self):
        """Charge les images croppées pour les racines et feuilles si elles existent"""
        if not self.current_dataset or not hasattr(self.current_dataset, 'crop_directory'):
            # Effacer les images si pas de dataset courant
            self.rootsTab.clearImage()
            self.leavesTab.clearImage()
            return
        
        for analysis_type in ["roots", "leaves"]:
            crop_loaded = False
            
            if (analysis_type in self.current_dataset.crop_directory and 
                self.current_dataset.crop_directory[analysis_type] and 
                os.path.exists(self.current_dataset.crop_directory[analysis_type])):
                
                cropped_name = replace_file_extension(self.current_image_name, new_extension='png')
                cropped_path = os.path.join(self.current_dataset.crop_directory[analysis_type], cropped_name)
                if not os.path.exists(cropped_path):
                    cropped_path = os.path.join(self.current_dataset.crop_directory[analysis_type], self.current_image_name)
                
                if os.path.exists(cropped_path):
                    cropped_image = cv2.cvtColor(image_read(cropped_path), cv2.COLOR_BGR2RGB)
                    
                    # Utiliser setImageArray au lieu de setImagePixmap pour préserver l'array
                    if analysis_type == "roots":
                        self.rootsTab.setImageArray(cropped_image)
                    else:
                        self.leavesTab.setImageArray(cropped_image)
                    crop_loaded = True
            
            # Si aucun crop n'a été chargé pour ce type, effacer l'affichage
            if not crop_loaded:
                if analysis_type == "roots":
                    self.rootsTab.clearImage()
                else:
                    self.leavesTab.clearImage()
    
    def updateAllSelections(self):
        """Met à jour les sélections affichées dans tous les onglets"""
        self.originalTab.setSelectionCoords(self.analysis_configs[self.current_analysis_type].selection_coords)
        self.rootsTab.setSelectionCoords(self.analysis_configs["roots"].selection_coords)
        self.leavesTab.setSelectionCoords(self.analysis_configs["leaves"].selection_coords)
        self.updateSelectionInfo()
    
    def openColorDialog(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.rgb_color = [color.red(), color.green(), color.blue()]
            self.colorChoiceButton.setStyleSheet(f"background-color: rgb({self.rgb_color[0]}, {self.rgb_color[1]}, {self.rgb_color[2]})")
            self.originalTab.changeColorSelection(self.rgb_color)
    
    def activateRectangleSelection(self):
        self.activeSelection = self.imageSelectionToolButton.isChecked()
        if not self.activeSelection:
            # Effacer seulement la sélection de l'onglet actuel
            current_widget = self.tabWidget.currentWidget()
            if self.current_tab_index == 0:
                current_widget.clearSelection()
            
            # Désactiver les boutons de confirmation
            self.confirmRootsSelectionButton.setEnabled(False)
            self.confirmLeavesSelectionButton.setEnabled(False)
            
            self.updateSelectionInfo()
    
    def onSelectionFinished(self, original_coords):
        """Appelée quand une sélection est terminée"""
        # Déterminer dans quel onglet la sélection a été faite
        current_index = self.tabWidget.currentIndex()
        self.temp_selection_coords = original_coords
        
        # Activer les boutons de confirmation
        self.confirmRootsSelectionButton.setEnabled(True)
        self.confirmLeavesSelectionButton.setEnabled(True)
        
        self.updateSelectionInfo()
    
    def updateSelectionInfo(self):
        """Met à jour les informations de sélection affichées"""
        current_index = self.tabWidget.currentIndex()
        
        # Afficher d'abord la sélection temporaire si elle existe
        if self.temp_selection_coords:
            coords = self.temp_selection_coords
            info_prefix = "Temporary Selection:"
            info_text = f"{info_prefix}\nX: {coords.x()}, Y: {coords.y()}\nW: {coords.width()}, H: {coords.height()}\n\nClick 'Confirm' to apply"
        else:
            # Sinon afficher la sélection confirmée selon l'onglet
            if current_index == 0:  # Original tab
                coords = self.analysis_configs[self.current_analysis_type].selection_coords
                info_prefix = f"{self.current_analysis_type.title()} Selection:"
            elif current_index == 1:  # Roots tab
                coords = self.analysis_configs["roots"].selection_coords
                info_prefix = "Roots Selection:"
            elif current_index == 2:  # Leaves tab
                coords = self.analysis_configs["leaves"].selection_coords
                info_prefix = "Leaves Selection:"
            else:
                coords = None
                info_prefix = "Selection:"
            
            if coords:
                info_text = f"{info_prefix}\nX: {coords.x()}, Y: {coords.y()}\nW: {coords.width()}, H: {coords.height()}"
            else:
                info_text = f"{info_prefix}\nNo selection"
        
        self.selectionInfoLabel.setText(info_text)
    
    def cropAllDatasets(self, analysis_type):
        """Crop toutes les images des datasets sélectionnés pour un type d'analyse"""
        selected_datasets = self.getSelectedDatasets()
        if not selected_datasets:
            QMessageBox.warning(self, "No Selection", "No datasets selected for processing.")
            return
        
        config = self.analysis_configs[analysis_type]
        
        if not config.selection_coords:
            # Utiliser l'image entière si pas de sélection
            if self.current_image_array is not None:
                height, width = self.current_image_array.shape[:2]
                config.selection_coords = QRect(0, 0, width, height)
                print(f"No selection for {analysis_type}, using full image")
            else:
                print(f"No selection and no image loaded for {analysis_type}")
                return
        
        if not self.base_output_directory:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
        
        # Créer les répertoires de sortie si nécessaire
        self.createAllOutputDirectories()
        
        # Créer et afficher la fenêtre de progression
        progress_window = ProgressWindow(self, crop_only=True)
        progress_window.start_processing(self, selected_datasets, [analysis_type])
        if PYQT_VERSION == 6:
            progress_window.exec()
        else:
            progress_window.exec_()
        
        # Recharger l'image courante pour afficher la version croppée
        self.loadCroppedImages()
        print(f"All {analysis_type} images cropped successfully for selected datasets!")
    
    
    def openThresholdWindow(self, analysis_type):
        """Ouvre la fenêtre de seuillage pour un type d'analyse"""
        config = self.analysis_configs[analysis_type]
        
        work_image = None
        work_image_source = None
        if analysis_type == "roots":
            work_image = self.rootsTab.getDisplayedImage()
            work_image_source = "roots"
        elif analysis_type == "leaves":
            work_image = self.leavesTab.getDisplayedImage()
            work_image_source = "leaves"
        
        if work_image is None:
            work_image = self.originalTab.getDisplayedImage()
            work_image_source = "original"
        
        if work_image is None:
            print(f"No image available for {analysis_type} analysis")
            return None
        
        # Normaliser l'ordre des canaux pour l'affichage dans la fenêtre de seuillage.
        # - Les images croppées roots/leaves sont chargées en RGB (loadCroppedImages()).
        # - L'image "original" dépend du format : cv2.imread -> BGR (non-TIFF), tifffile -> généralement déjà RGB.
        if work_image_source == "original":
            path_hint = getattr(self, "current_image_name", None)
            work_image = ensure_rgb(work_image, path_hint)
        
        # Créer la fenêtre de seuillage avec les paramètres spécifiques
        self.threshold_window = EnhancedOptionsWindow(
            f"{analysis_type.title()} Analysis Parameters",
            self.icons_directory,
            parent_app=self,
            init_params=config.threshold_params
        )
        
        # Configurer avec les paramètres existants
        self.threshold_window.setImageData(work_image, None)
        self.setupThresholdWindow(analysis_type)
        
        # Stocker le type d'analyse pour le callback
        self.current_threshold_type = analysis_type
        
        self.threshold_window.show()
    
    def setupThresholdWindow(self, analysis_type):
        """Configure la fenêtre de seuillage avec les paramètres existants"""
        config = self.analysis_configs[analysis_type]
        params = config.threshold_params
        
        # Configurer les sliders avec les valeurs sauvegardées
        self.threshold_window.red_lower_value = params['red_range'][0]
        self.threshold_window.red_upper_value = params['red_range'][1]
        self.threshold_window.green_lower_value = params['green_range'][0]
        self.threshold_window.green_upper_value = params['green_range'][1]
        self.threshold_window.blue_lower_value = params['blue_range'][0]
        self.threshold_window.blue_upper_value = params['blue_range'][1]
        
        # Configurer les sliders visuels
        self.threshold_window.red_slider.setLow(params['red_range'][0])
        self.threshold_window.red_slider.setHigh(params['red_range'][1])
        self.threshold_window.green_slider.setLow(params['green_range'][0])
        self.threshold_window.green_slider.setHigh(params['green_range'][1])
        self.threshold_window.blue_slider.setLow(params['blue_range'][0])
        self.threshold_window.blue_slider.setHigh(params['blue_range'][1])
        
        # Configurer les checkboxes
        self.threshold_window.red_inverted_values = params['red_invert']
        self.threshold_window.green_inverted_values = params['green_invert']
        self.threshold_window.blue_inverted_values = params['blue_invert']
        
        self.threshold_window.red_invert_checkbox.setChecked(params['red_invert'])
        self.threshold_window.green_invert_checkbox.setChecked(params['green_invert'])
        self.threshold_window.blue_invert_checkbox.setChecked(params['blue_invert'])
        
        # Configurer l'aire minimale des composantes connexes
        self.threshold_window.min_connected_components_area = params['min_connected_components_area']
        
        # Configurer la taille d'objet minimum
        self.threshold_window.min_object_size = params['min_object_size']
        self.threshold_window.object_size_entry.setText(str(params['min_object_size']))
        
        # Mettre à jour les labels
        self.threshold_window.red_min_label.setText(str(params['red_range'][0]))
        self.threshold_window.red_max_label.setText(str(params['red_range'][1]))
        self.threshold_window.green_min_label.setText(str(params['green_range'][0]))
        self.threshold_window.green_max_label.setText(str(params['green_range'][1]))
        self.threshold_window.blue_min_label.setText(str(params['blue_range'][0]))
        self.threshold_window.blue_max_label.setText(str(params['blue_range'][1]))
        
        # Déclencher la mise à jour de la prévisualisation
        self.threshold_window._updatePreview()
    
    def setColorThresholdValues(self, red_interval, blue_interval, green_interval, invert_red=False, invert_blue=False, invert_green=False):
        """Callback appelé par la fenêtre de seuillage"""
        analysis_type = getattr(self, 'current_threshold_type', 'roots')
        config = self.analysis_configs[analysis_type]
        
        # Sauvegarder les nouveaux paramètres
        config.threshold_params.update({
            'red_range': tuple(red_interval),
            'green_range': tuple(green_interval),
            'blue_range': tuple(blue_interval),
            'red_invert': invert_red,
            'green_invert': invert_green,
            'blue_invert': invert_blue,
            'keep_max_component': getattr(self.threshold_window, 'keep_max_component', False),
            'min_connected_components_area': getattr(self.threshold_window, 'min_connected_components_area', 0),
            'min_object_size': getattr(self.threshold_window, 'min_object_size', 0),
            'kernel_size': getattr(self.threshold_window, 'kernel_size', 5),
            'kernel_shape': getattr(self.threshold_window, 'kernel_shape_value', cv2.MORPH_RECT),
            'fusion_masks': getattr(self.threshold_window, 'fusion_previous_masks', True)
        })
        
        print(f"{analysis_type.title()} threshold parameters updated:")
        print(f"Red: {red_interval} (inverted: {invert_red})")
        print(f"Green: {green_interval} (inverted: {invert_green})")
        print(f"Blue: {blue_interval} (inverted: {invert_blue})")
    
    def getCleaningParameters(self):
        """Get parameters values for cleaning roots masks"""
        config = self.analysis_configs['roots']
        params = config.threshold_params
        
        self.cleaning_mask_parameters['closing_radius'] = params['kernel_size']
        self.cleaning_mask_parameters['closing_shape'] = params['kernel_shape']
        self.cleaning_mask_parameters['min_branch_length'] = params['min_branch_length']
        self.cleaning_mask_parameters['keep_max_component'] = params['keep_max_component']
        self.cleaning_mask_parameters['min_connected_components_area'] = params['min_connected_components_area']
        self.cleaning_mask_parameters['min_object_size'] = params['min_object_size']
        self.cleaning_mask_parameters['line_thickness'] = 5
        self.cleaning_mask_parameters['temporal_merge'] = params['fusion_masks']
        self.cleaning_mask_parameters['connect_objects'] = True
    
    def openRootArchitecture(self):
        """Ouverture de la fenêtre d'analyse racinaire"""
        self.getCleaningParameters()
        
        if self.root_arch_window is None or not self.root_arch_window.isVisible():
            self.root_arch_window = RootArchitectureWindow(self, init_params=self.cleaning_mask_parameters, 
                                                            datasets=self.datasets, current_dir=self.current_directory, 
                                                            output_dir=self.base_output_directory, screen_size=self.screen_size)
            self.root_arch_window.show()
        else:
            self.root_arch_window.raise_()
            self.root_arch_window.activateWindow()
    
    def exportSegmentedImages(self):
        """Exporte les images segmentées pour le type d'analyse courant"""
        analysis_type = getattr(self, 'current_threshold_type', 'roots')
        selected_datasets = self.getSelectedDatasets()
        
        if not selected_datasets:
            QMessageBox.warning(self, "No Selection", "No datasets selected for processing.")
            return
        
        if not self.base_output_directory:
            print(f"No output directory for {analysis_type} segmentation")
            return
        
        # Créer les répertoires de sortie si nécessaire
        self.createAllOutputDirectories()
        
        # Créer et afficher la fenêtre de progression pour la segmentation
        progress_window = ProgressWindow(self, segment_only=True)
        progress_window.start_processing(self, selected_datasets, [analysis_type])
        if PYQT_VERSION == 6:
            progress_window.exec()
        else:
            progress_window.exec_()
    
    
    def createGlobalCSV(self, analysis_type):
        """Crée un CSV global avec les données de tous les datasets sélectionnés"""
        selected_datasets = self.getSelectedDatasets()
        
        global_data = {
            "Dataset": [],
            "Image name": [], 
            "Analysis type": [],
            "Pixel count": [], 
            "Convex area": [], 
            "Modality": [], 
            "Day": []
        }
        
        # Collecter toutes les données des datasets sélectionnés
        for dataset_name in selected_datasets:
            if dataset_name not in self.datasets:
                continue
                
            dataset_info = self.datasets[dataset_name]
            csv_path = os.path.join(
                os.path.dirname(dataset_info.segmented_directory[analysis_type]),
                f"{dataset_name}_{analysis_type}_analysis.csv"
            )
            
            if os.path.exists(csv_path):
                # Lire le CSV existant et ajouter au global
                try:
                    df = pd.read_csv(csv_path, sep=';')
                    for _, row in df.iterrows():
                        for key in global_data.keys():
                            if key in row:
                                global_data[key].append(row[key])
                            else:
                                global_data[key].append('NaN')
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    # Alternative sans pandas
                    try:
                        with open(csv_path, 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 1:
                                headers = lines[0].strip().split(';')
                                for line in lines[1:]:
                                    values = line.strip().split(';')
                                    for i, key in enumerate(global_data.keys()):
                                        if i < len(values):
                                            global_data[key].append(values[i])
                                        else:
                                            global_data[key].append('NaN')
                    except Exception as e2:
                        print(f"Error reading CSV manually: {e2}")
        
        # Sauvegarder le CSV global
        global_csv_path = os.path.join(self.base_output_directory, f"global_{analysis_type}_analysis.csv")
        export_convex_area(global_csv_path, global_data, csv_separator=';')
        
        if os.path.exists(global_csv_path):
            print(f"Global {analysis_type} data exported to: {global_csv_path}")
    
    def createDatasetGraph(self):
        dataset_name = self.current_dataset
        # Vider la figure précédente
        self.figure.clear()
        
        idx = 0
        graphs_created = 0
        for analysis_type in ['roots', 'leaves']:
            current_res = self.analysis_dict.get(analysis_type, {}).get(dataset_name, None)
            print(f"=== Analyse: {analysis_type} ===")
            print(f"Dataset: {dataset_name} (type: {type(dataset_name)})")
            print(f"current_res: {current_res}")
            if current_res and isinstance(current_res, dict):
                print(f"Clés disponibles: {list(current_res.keys())}")
                for key in ["Day", "Pixel count", "Modality", "Image name"]:
                    if key in current_res:
                        val = current_res[key]
                        print(f"  {key}: {len(val) if hasattr(val, '__len__') else 'N/A'} éléments")
                        if hasattr(val, '__len__') and len(val) > 0:
                            print(f"    Échantillon: {val[:3] if len(val) >= 3 else val}")
                    else:
                        print(f"  {key}: ABSENT")
            if current_res:
                idx += 1
                data = pd.DataFrame()
                data["Day"] = pd.to_numeric(current_res["Day"], errors="coerce")
                data["Pixel count"] = pd.to_numeric(current_res["Pixel count"], errors="coerce")
                data["Convex area"] = pd.to_numeric(current_res["Convex area"], errors="coerce")
                
                # Gérer la modalité - si elle n'existe pas, créer une modalité par défaut
                if "Modality" in current_res and current_res["Modality"]:
                    data["Modality"] = current_res["Modality"]
                    print(f"Modalités trouvées: {set(current_res['Modality'])}")
                else:
                    print(f"⚠ Pas de colonne Modality, création d'une modalité par défaut")
                    data["Modality"] = ["Default"] * len(data)
                print(f"DataFrame créé: {data.shape}")
                data = data.dropna(subset=["Day", "Pixel count", "Convex area"])
                print(f"Après nettoyage: {data.shape}")
                
                if len(data) == 0:
                    print(f"⚠ Aucune donnée valide pour {analysis_type} après nettoyage")
                    continue
                
                # Calcul des moyennes et des erreurs standards
                grouped = data.groupby(["Modality", "Day"]).agg(
                    Mean=("Pixel count", "mean"),
                    se=("Pixel count", lambda x: x.std() / (len(x) ** 0.5))
                ).reset_index()
                print(f"✓ Groupement réussi: {len(grouped)} points, modalités: {grouped['Modality'].unique()}")
                
                # Initialiser le graphique
                ax = self.figure.add_subplot(2, 1, idx)
                
                # Tracer une courbe pour chaque moda
                modas = grouped["Modality"].unique()
                for moda in modas:
                    subset = grouped[grouped["Modality"] == moda]
                    ax.errorbar(
                        subset["Day"],
                        subset["Mean"],
                        yerr=subset["se"],
                        label=f"Modality {moda}",
                        capsize=3,
                        marker="o"
                    )
                
                # Configuration du graphique
                ax.set_title(f"Cinétique de 'Pixel count' en fonction des jours", 
                            fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel("Day", fontsize=12, fontweight='bold')
                ax.set_ylabel("Pixels", fontsize=12, fontweight='bold')
                # Améliorer la légende avec style Seaborn
                legend = ax.legend(title="Modality", title_fontsize=11, fontsize=10, 
                                 loc='upper left', frameon=True, fancybox=True, shadow=True)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(0.9)
                # Grille élégante avec style Seaborn
                ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)
                ax.set_facecolor('#fafafa')
                
                # Améliorer les ticks et spines
                ax.tick_params(axis='both', which='major', labelsize=10)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)
                    spine.set_color('#cccccc')
                
                graphs_created += 1
                print(f"✓ Graphique {analysis_type} créé avec succès")
        
        # Affichage final après tous les graphiques
        print(f"Nombre de graphiques créés: {graphs_created}")
        if graphs_created > 0:
            self.figure.tight_layout(pad=2.0)
            self.canvas.draw()
            print(f"=== FIN: {graphs_created} graphique(s) affiché(s) ===")
        else:
            # Afficher un message si aucune donnée
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f'Aucune donnée disponible\npour le dataset: {dataset_name}', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            print("=== FIN: Aucune donnée - message d'information affiché ===")
    
    def applyColorThreshold(self, image, params):
        """Applique le seuillage RGB avec les paramètres donnés"""
        # Créer des intervalles selon les paramètres d'inversion
        def create_intervals(range_param, invert):
            low, high = range_param
            if invert:
                if low == 0 and high == 255:
                    return []
                elif low == 0:
                    return [[high, 255]]
                elif high == 255:
                    return [[0, low]]
                else:
                    return [[0, low], [high, 255]]
            else:
                return [range_param]
        
        red_intervals = create_intervals(params['red_range'], params['red_invert'])
        green_intervals = create_intervals(params['green_range'], params['green_invert'])
        blue_intervals = create_intervals(params['blue_range'], params['blue_invert'])
        
        # Appliquer le seuillage
        mask = np.zeros([image.shape[0], image.shape[1]], dtype='uint8')
        
        if red_intervals and green_intervals and blue_intervals:
            for red_int in red_intervals:
                for green_int in green_intervals:
                    for blue_int in blue_intervals:
                        lower_bound = np.array([red_int[0], green_int[0], blue_int[0]], dtype='uint8')
                        upper_bound = np.array([red_int[1], green_int[1], blue_int[1]], dtype='uint8')
                        mask = cv2.bitwise_or(mask, cv2.inRange(image, lower_bound, upper_bound))
        
        return mask
    
    def previousImage(self):
        if self.n_images_loaded > 0:
            self.idx_current_image = (self.idx_current_image - 1) % self.n_images_loaded
            self.displayImageIndex(self.idx_current_image)
            self.updateImageIndexLabel()
    
    def nextImage(self):
        if self.n_images_loaded > 0:
            self.idx_current_image = (self.idx_current_image + 1) % self.n_images_loaded
            self.displayImageIndex(self.idx_current_image)
            self.updateImageIndexLabel()
    
    def updateImageIndexLabel(self):
        if self.n_images_loaded > 0:
            dataset_name = self.current_dataset.name if self.current_dataset else "Unknown"
            self.imageIndexLabel.setText(f"{dataset_name}: {self.idx_current_image + 1} / {self.n_images_loaded}")
        else:
            self.imageIndexLabel.setText("Image: 0 / 0")



class ApplicationWindow(QMainWindow):
    def __init__(self, screen=None):
        super(ApplicationWindow, self).__init__()
        self.screen = screen
        self.screen_size = [self.screen.size().width(), self.screen.size().height()]
        self.max_main_window_size = [1800, 1300]
        self.main_geometry = [0, 0, min(int(self.screen_size[0] * 0.9), self.max_main_window_size[0]), min(int(self.screen_size[1] * 0.9), self.max_main_window_size[1])]
        self.main_geometry[0] = min(100, self.screen_size[0] - self.main_geometry[2])
        self.main_geometry[1] = min(100, self.screen_size[1] - self.main_geometry[3])
        
        self.setWindowTitle("Dataset Analysis: Roots & Leaves System")
        self.setGeometry(self.main_geometry[0], self.main_geometry[1], self.main_geometry[2], self.main_geometry[3])
        
        # Widget central
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.icons_directory = os.path.join(self.current_dir, "icons")
        self.mainWidget = App("main", self.current_dir, screen_size=self.screen_size, parent=self)
        self.setCentralWidget(self.mainWidget)
        
        # Icons
        self.icon_menu_file_name = "IconRTT.png"
        self.icon_menu_file_path = os.path.join(self.icons_directory, self.icon_menu_file_name)
        self.setWindowIcon(QIcon(self.icon_menu_file_path))
        
        # Menu
        self._createMenuBar()
    
    def _createMenuBar(self):
        menubar = self.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu('File')
        
        # Sélection des dossiers
        select_datasets = QAction('Select Datasets Folder...', self)
        select_datasets.setShortcut('Ctrl+O')
        select_datasets.triggered.connect(self.mainWidget.openBaseDirectoryDialog)
        file_menu.addAction(select_datasets)
        
        select_output = QAction('Select Output Folder...', self)
        select_output.setShortcut('Ctrl+S')
        select_output.triggered.connect(self.mainWidget.openOutputDirectoryDialog)
        file_menu.addAction(select_output)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Analysis
        analysis_menu = menubar.addMenu('Analysis')
        
        roots_threshold = QAction('Roots Threshold...', self)
        roots_threshold.triggered.connect(lambda: self.mainWidget.openThresholdWindow("roots"))
        analysis_menu.addAction(roots_threshold)
        
        leaves_threshold = QAction('Leaves Threshold...', self)
        leaves_threshold.triggered.connect(lambda: self.mainWidget.openThresholdWindow("leaves"))
        analysis_menu.addAction(leaves_threshold)
        
        analysis_menu.addSeparator()
        
        # Actions de crop
        crop_roots = QAction('Crop All Roots', self)
        crop_roots.triggered.connect(lambda: self.mainWidget.cropAllDatasets("roots"))
        analysis_menu.addAction(crop_roots)
        
        crop_leaves = QAction('Crop All Leaves', self)
        crop_leaves.triggered.connect(lambda: self.mainWidget.cropAllDatasets("leaves"))
        analysis_menu.addAction(crop_leaves)
        
        analysis_menu.addSeparator()
        
        # Analyse et visualisation du système racinaire
        root_arch_action = QAction('Roots Architecture', self)
        root_arch_action.setShortcut('Ctrl+R')
        root_arch_action.setStatusTip('Open root architecture analysis')
        root_arch_action.triggered.connect(self.mainWidget.openRootArchitecture)
        analysis_menu.addAction(root_arch_action)
        
        # Menu View
        view_menu = menubar.addMenu('View')
        
        prev_image = QAction('Previous Image', self)
        prev_image.setShortcut('Left')
        prev_image.triggered.connect(self.mainWidget.previousImage)
        view_menu.addAction(prev_image)
        
        next_image = QAction('Next Image', self)
        next_image.setShortcut('Right')
        next_image.triggered.connect(self.mainWidget.nextImage)
        view_menu.addAction(next_image)


def main():
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    
    window = ApplicationWindow(screen=screen)
    window.show()
    if PYQT_VERSION == 6:
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())


if __name__ == '__main__':
    if PYQT_AVAILABLE:
        main()