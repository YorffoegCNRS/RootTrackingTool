import cv2, os, sys
import numpy as np

from skimage.morphology import remove_small_objects

PYQT_AVAILABLE = False
PYQT_VERSION = None

try:
    from PyQt6.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QComboBox, QSlider, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QSizePolicy, QListWidget, QListWidgetItem
    from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QImage, QPainter, QIntValidator
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
except:
    try:
        from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
        from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QComboBox, QSlider, QVBoxLayout, QHBoxLayout, QFrame, QCheckBox, QSizePolicy, QListWidget, QListWidgetItem
        from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QImage, QPainter, QIntValidator
        PYQT_AVAILABLE = True
        PYQT_VERSION = 5
    except:
        print("This script requires PyQt5 or PyQt6 to run. Neither of these versions was found!")



class RangeSlider(QSlider):
    """Range slider personnalisé"""
    sliderMoved = pyqtSignal(int, int)
    
    def __init__(self, *args):
        super(RangeSlider, self).__init__(*args)
        self._low = self.minimum()
        self._high = self.maximum()
        self.pressed_control = None
        self.hover_control = None
        self.click_offset = 0
        self.active_slider = 0
    
    def low(self):
        return self._low

    def setLow(self, low: int):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high: int):
        self._high = high
        self.update()
    
    def paintEvent(self, event):
        # Implémentation simplifiée du paintEvent
        painter = QPainter(self)
        
        # Dessiner la barre de base
        painter.fillRect(self.rect(), Qt.GlobalColor.lightGray if PYQT_VERSION == 6 else Qt.lightGray)
        
        # Calculer les positions des handles
        total_width = self.width() - 20
        low_pos = int((self._low / 255) * total_width) + 10
        high_pos = int((self._high / 255) * total_width) + 10
        
        # Dessiner la zone sélectionnée
        selected_rect = QRect(low_pos, 5, high_pos - low_pos, self.height() - 10)
        painter.fillRect(selected_rect, Qt.GlobalColor.blue if PYQT_VERSION == 6 else Qt.blue)
        
        # Dessiner les handles
        painter.fillRect(low_pos - 5, 2, 10, self.height() - 4, Qt.GlobalColor.darkBlue if PYQT_VERSION == 6 else Qt.darkBlue)
        painter.fillRect(high_pos - 5, 2, 10, self.height() - 4, Qt.GlobalColor.darkBlue if PYQT_VERSION == 6 else Qt.darkBlue)
    
    def mousePressEvent(self, event):
        total_width = self.width() - 20
        click_pos = event.x() - 10
        value = int((click_pos / total_width) * 255)
        
        low_pos = int((self._low / 255) * total_width)
        high_pos = int((self._high / 255) * total_width)
        
        # Déterminer quel handle est le plus proche
        if abs(click_pos - low_pos) < abs(click_pos - high_pos):
            self._low = max(0, min(255, value))
            self.active_slider = 0
        else:
            self._high = max(0, min(255, value))
            self.active_slider = 1
        
        self.update()
        self.sliderMoved.emit(self._low, self._high)
    
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            total_width = self.width() - 20
            click_pos = event.x() - 10
            value = max(0, min(255, int((click_pos / total_width) * 255)))
            
            if self.active_slider == 0:
                self._low = min(value, self._high - 1)
            else:
                self._high = max(value, self._low + 1)
            
            self.update()
            self.sliderMoved.emit(self._low, self._high)



class DatasetSelectionWidget(QFrame):
    """
    Widget réutilisable pour:
    - afficher la liste des datasets (QListWidget)
    - cocher/décocher des datasets pour le traitement
    - choisir un dataset "Viewing" (clic item)
    - select all / deselect all
    """
    
    # dataset cliqué => "viewing"
    dataset_view_requested = pyqtSignal(str)
    
    # la sélection (checkbox) a changé => utile pour activer des boutons, batch, etc.
    selection_changed = pyqtSignal()
    
    def __init__(self, parent=None, title="Datasets:"):
        super().__init__(parent)
        
        self.datasets = {}          # dict[str, DatasetInfo]
        self._current_view = None   # str | None
        self._updating_ui = False
        self._locked = False  # lock dataset switching during analysis
        
        
        self.setFrameStyle(QFrame.Shape.StyledPanel if PYQT_VERSION == 6 else QFrame.StyledPanel)
        
        layout = QVBoxLayout(self)
        
        # Titre
        self.datasets_title = QLabel(title)
        self.datasets_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        layout.addWidget(self.datasets_title)
        
        # Boutons select/deselect
        btns = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        btns.addWidget(self.select_all_btn)
        btns.addWidget(self.deselect_all_btn)
        layout.addLayout(btns)
        
        # Liste
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.list_widget)
        
        # Labels statut
        self.current_view_label = QLabel("Viewing: -")
        self.selected_count_label = QLabel("Selected: 0/0")
        layout.addWidget(self.current_view_label)
        layout.addWidget(self.selected_count_label)
    
    # -------------------------
    # Helpers Qt compat
    # -------------------------
    def _checked(self):
        return Qt.CheckState.Checked if PYQT_VERSION == 6 else Qt.Checked
    
    def _unchecked(self):
        return Qt.CheckState.Unchecked if PYQT_VERSION == 6 else Qt.Unchecked
    
    def _user_checkable_flag(self):
        return Qt.ItemFlag.ItemIsUserCheckable if PYQT_VERSION == 6 else Qt.ItemIsUserCheckable
    
    # -------------------------
    # API publique
    # -------------------------
    def set_datasets(self, datasets: dict, select_first_for_view=True):
        """
        datasets: dict[str, DatasetInfo] (DatasetInfo doit avoir .selected et .name/.path)
        """
        self.datasets = datasets or {}
        self.refresh(select_first_for_view=select_first_for_view)
    
    def refresh(self, select_first_for_view=True):
        """Reconstruit la liste depuis self.datasets."""
        self._updating_ui = True
        try:
            self.list_widget.clear()
            
            for ds_name in sorted(self.datasets.keys()):
                item = QListWidgetItem(ds_name)
                item.setFlags(item.flags() | self._user_checkable_flag())
                item.setCheckState(self._checked() if getattr(self.datasets[ds_name], "selected", True) else self._unchecked())
                self.list_widget.addItem(item)
            
            if select_first_for_view and self.list_widget.count() > 0:
                self.list_widget.setCurrentRow(0)
                first_name = self.list_widget.item(0).text()
                self.set_viewing_dataset(first_name, emit_signal=True)
            else:
                # garder l'affichage cohérent
                self._update_selected_count()
        finally:
            self._updating_ui = False
    
    def get_selected_dataset_names(self):
        return [name for name, ds in self.datasets.items() if getattr(ds, "selected", False)]
    
    def set_viewing_dataset(self, dataset_name: str, emit_signal=False):
        """Change le dataset 'Viewing' (sans modifier selected).
        
        IMPORTANT: si ce changement est déclenché programmaticalement (ex: revert pendant une analyse),
        on resynchronise la sélection visuelle dans la QListWidget pour éviter les états "bloqués".
        """
        if dataset_name not in self.datasets:
            return
        
        # Resync list selection to match the requested viewing dataset
        try:
            # Find row
            row = None
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).text() == dataset_name:
                    row = i
                    break
            if row is not None and row >= 0:
                self._updating_ui = True
                try:
                    self.list_widget.setCurrentRow(row)
                finally:
                    self._updating_ui = False
        except Exception:
            pass
        
        self._current_view = dataset_name
        self.current_view_label.setText(f"Viewing: {dataset_name}")
        self._update_selected_count()
        if emit_signal:
            self.dataset_view_requested.emit(dataset_name)
    
    
    def set_locked(self, locked: bool):
        """Enable/disable any user interaction that can change the viewing dataset or the batch selection."""
        self._locked = bool(locked)
        enabled = not self._locked
        
        # Gray out controls
        try:
            self.select_all_btn.setEnabled(enabled)
            self.deselect_all_btn.setEnabled(enabled)
        except Exception:
            pass
        
        try:
            self.list_widget.setEnabled(enabled)
        except Exception:
            pass
        
        # Optional: make it visually obvious even if the OS/theme doesn't gray much
        try:
            self.setProperty("locked", self._locked)
            self.style().unpolish(self)
            self.style().polish(self)
        except Exception:
            pass
    
    def current_viewing_dataset(self):
        return self._current_view
    
    def select_all(self):
        if getattr(self, '_locked', False):
            return
        self._set_all_checkstates(True)
    
    def deselect_all(self):
        if getattr(self, '_locked', False):
            return
        self._set_all_checkstates(False)
    
    # -------------------------
    # Internals
    # -------------------------
    def _set_all_checkstates(self, state: bool):
        if getattr(self, '_locked', False):
            return
        self._updating_ui = True
        try:
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                item.setCheckState(self._checked() if state else self._unchecked())
                ds_name = item.text()
                if ds_name in self.datasets:
                    self.datasets[ds_name].selected = state
            self._update_selected_count()
        finally:
            self._updating_ui = False
        
        self.selection_changed.emit()
    
    def _update_selected_count(self):
        total = len(self.datasets)
        selected = sum(1 for ds in self.datasets.values() if getattr(ds, "selected", False))
        self.selected_count_label.setText(f"Selected: {selected}/{total}")
    
    def _on_item_clicked(self, item):
        if getattr(self, '_locked', False):
            return
        if not item:
            return
        ds_name = item.text()
        self.set_viewing_dataset(ds_name, emit_signal=True)
    
    def _on_item_changed(self, item):
        if getattr(self, '_locked', False):
            return
        if self._updating_ui or not item:
            return
        
        ds_name = item.text()
        if ds_name in self.datasets:
            self.datasets[ds_name].selected = (item.checkState() == self._checked())
        
        self._update_selected_count()
        self.selection_changed.emit()





class PreviewWidget(QLabel):
    """Widget pour afficher la prévisualisation du seuillage"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Supprimer les contraintes de taille fixes
        self.setMinimumSize(400, 300)  # Taille minimum raisonnable
        # Ne pas définir de taille maximum pour permettre l'expansion
        self.setScaledContents(False)  # Important pour gérer nous-mêmes le scaling
        self.setStyleSheet("border: 1px solid gray;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding if PYQT_VERSION == 6 else QSizePolicy.Expanding, QSizePolicy.Policy.Expanding if PYQT_VERSION == 6 else QSizePolicy.Expanding)
        
        # Variables
        self.original_image = None
        self.selection_coords = None
        self.current_pixmap = None
        
    def setOriginalImage(self, image_array):
        """Définit l'image originale"""
        self.original_image = image_array
        
    def setSelectionCoords(self, coords):
        """Définit les coordonnées de sélection (QRect)"""
        self.selection_coords = coords
        
    def updatePreview(self, red_range, green_range, blue_range, red_invert, green_invert, blue_invert, closing_kernel=None, keep_max_component_only=False, object_size=0):
        """Met à jour la prévisualisation avec les nouveaux paramètres de seuillage"""
        if self.original_image is None:
            return
            
        # Extraire la zone sélectionnée si disponible
        if self.selection_coords is not None:
            x, y, w, h = self.selection_coords.x(), self.selection_coords.y(), self.selection_coords.width(), self.selection_coords.height()
            # S'assurer que les coordonnées sont dans les limites
            h_img, w_img = self.original_image.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))
            work_image = self.original_image[y:y+h, x:x+w].copy()
        else:
            # Utiliser l'image entière si pas de sélection
            work_image = self.original_image.copy()
        
        # Appliquer le seuillage RGB
        mask = self.apply_rgb_threshold(
            work_image, red_range, green_range, blue_range, 
            red_invert, green_invert, blue_invert
        )
        segmented = (mask.astype(np.uint8) * 255)
        
        if closing_kernel is not None:
            segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, closing_kernel)
        
        if keep_max_component_only:
            retval, labels = cv2.connectedComponents(segmented)
            
            label_count = np.zeros(retval, dtype='int32')
            for i in range(1, retval):
                label_count[i] = np.sum(labels == i)
            
            max_label = np.argmax(label_count)
            segmented = np.array(labels == max_label, dtype='uint8') * 255
            
        
        # Appliquer la suppression de bruit
        if object_size > 0:
            segmented = remove_small_objects(np.array(segmented == 255, dtype='bool'), min_size=object_size)
            segmented = (segmented*255).astype('uint8')
        
        # Convertir en QPixmap et afficher
        red_segmented = np.zeros([segmented.shape[0], segmented.shape[1], 3], dtype='uint8')
        red_segmented[:, :, 0][segmented > 0] = 255
        self.displayThresholdedImage(red_segmented)
    
    def apply_rgb_threshold(self, image, red_range, green_range, blue_range, red_invert, green_invert, blue_invert):
        """Applique le seuillage RGB sur l'image"""
        # Créer les masques pour chaque canal
        red_mask = self.create_channel_mask(image[:,:,0], red_range, red_invert)
        green_mask = self.create_channel_mask(image[:,:,1], green_range, green_invert)
        blue_mask = self.create_channel_mask(image[:,:,2], blue_range, blue_invert)
        
        # Combiner les masques (ET logique)
        combined_mask = red_mask & green_mask & blue_mask
        
        # Créer l'image résultat
        # result = np.zeros_like(image)
        # result[combined_mask] = image[combined_mask]
        
        return combined_mask
    
    def create_channel_mask(self, channel, value_range, invert):
        """Crée un masque pour un canal de couleur"""
        low, high = value_range
        
        if invert:
            # Garder les valeurs HORS de l'intervalle
            mask = (channel < low) | (channel > high)
        else:
            # Garder les valeurs DANS l'intervalle
            mask = (channel >= low) & (channel <= high)
            
        return mask
    
    def displayThresholdedImage(self, image):
        """Affiche l'image seuillée dans le widget"""
        if image.size == 0:
            return
            
        # Convertir en QImage
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image, width, height, bytes_per_line, QImage.Format.Format_RGB888 if PYQT_VERSION == 6 else QImage.Format_RGB888)
        else:
            bytes_per_line = width
            q_image = QImage(image, width, height, bytes_per_line, QImage.Foramt.Format_Grayscale8 if PYQT_VERSION == 6 else QImage.Format_Grayscale8)
        
        # Convertir en QPixmap
        self.current_pixmap = QPixmap.fromImage(q_image)
        
        # Mettre à l'échelle et afficher
        self._updateScaledPixmap()
    
    def resizeEvent(self, event):
        """Redimensionne proprement le pixmap en gardant le ratio"""
        super().resizeEvent(event)
        self._updateScaledPixmap()

    def _updateScaledPixmap(self):
        """Met à jour l'affichage avec la bonne mise à l'échelle"""
        if self.current_pixmap is not None and not self.current_pixmap.isNull():
            # Calculer la mise à l'échelle en gardant les proportions
            widget_size = self.size()
            pixmap_size = self.current_pixmap.size()
            
            scale_factor = min(
                widget_size.width() / pixmap_size.width(),
                widget_size.height() / pixmap_size.height()
            )
            
            # Calculer la nouvelle taille
            new_width = int(pixmap_size.width() * scale_factor)
            new_height = int(pixmap_size.height() * scale_factor)
            
            # Redimensionner le pixmap
            scaled_pixmap = self.current_pixmap.scaled(
                new_width, new_height, 
                Qt.AspectRatioMode.KeepAspectRatio if PYQT_VERSION == 6 else Qt.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation if PYQT_VERSION == 6 else Qt.SmoothTransformation
            )
            
            # Centrer le pixmap dans le widget
            self.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
            self.setPixmap(scaled_pixmap)
    
    def paintEvent(self, event):
        """Dessine l'image centrée"""
        super().paintEvent(event)


class EnhancedOptionsWindow(QWidget):
    def __init__(self, window_title, icons_directory, parent_app=None, init_params=None, window_width=1400, window_height=900):
        super().__init__()
        self.window_title = window_title
        self.icons_directory = icons_directory
        self.parent_app = parent_app
        self.init_params = init_params or {}
        self.window_width = window_width
        self.window_height = window_height
        
        # Variables threshold initialisation
        self.red_lower_value = 0
        self.red_upper_value = 255
        self.green_lower_value = 0
        self.green_upper_value = 255
        self.blue_lower_value = 0
        self.blue_upper_value = 255
        self.red_inverted_values = self.init_params.get('red_invert', False)
        self.green_inverted_values = self.init_params.get('green_invert', False)
        self.blue_inverted_values = self.init_params.get('blue_invert', False)
        self.default_fusion_masks_state = self.init_params.get('fusion_masks', False)
        self.fusion_previous_masks = self.default_fusion_masks_state
        self.keep_max_component_only = self.init_params.get('keep_max_component', False)
        self.default_object_size = self.init_params.get('min_object_size', 800)
        self.max_object_size_value = 1e10
        self.min_object_size = self.default_object_size
        self.kernel_shape_dict = {"Rectangle":cv2.MORPH_RECT, "Cross":cv2.MORPH_CROSS, "Ellipse":cv2.MORPH_ELLIPSE}
        self.default_kernel_size = 0
        self.default_kernel_shape_index = 0
        self.default_kernel_shape_name = list(self.kernel_shape_dict.keys())[self.default_kernel_shape_index]
        self.kernel_size = (self.default_kernel_size, self.default_kernel_size)
        self.kernel_shape = self.default_kernel_shape_name
        self.kernel_shape_value = self.kernel_shape_dict[self.default_kernel_shape_name]
        self.closing_kernel = None
        
        # Interface
        self.setWindowTitle(window_title)
        self.setGeometry(200, 200, window_width, window_height)
        self.setMinimumSize(1000, 700)  # Taille minimum de la fenêtre
        self._createInterface()
    
    def _createInterface(self):
        """Crée l'interface de la fenêtre"""
        main_layout = QHBoxLayout(self)
        
        # Panel de gauche (contrôles) - taille fixe
        left_panel = self._createControlPanel()
        left_panel.setFixedWidth(400)  # Largeur fixe pour les contrôles
        main_layout.addWidget(left_panel)
        
        # Panel de droite (prévisualisation) - extensible
        right_panel = self._createPreviewPanel()
        main_layout.addWidget(right_panel)
        
        # Proportions : contrôles fixes, prévisualisation extensible
        main_layout.setStretchFactor(left_panel, 0)  # Pas d'étirement
        main_layout.setStretchFactor(right_panel, 1)  # S'étire pour remplir l'espace
    
    def _createControlPanel(self):
        """Crée le panel de contrôles"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel if PYQT_VERSION == 6 else QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Titre
        title = QLabel("RGB Threshold Settings")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(title)
        
        # Contrôles Rouge
        red_group = self._createColorGroup("Red", "red")
        layout.addWidget(red_group)
        
        # Contrôles Vert
        green_group = self._createColorGroup("Green", "green")
        layout.addWidget(green_group)
        
        # Contrôles Bleu
        blue_group = self._createColorGroup("Blue", "blue")
        layout.addWidget(blue_group)
        
        # Remove outliers part
        remove_outliers_title = QLabel("Remove Outliers")
        remove_outliers_title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        remove_outliers_title.setAlignment(Qt.AlignmentFlag.AlignLeft if PYQT_VERSION == 6 else Qt.AlignLeft)
        layout.addWidget(remove_outliers_title)
        
        self.remove_outliers_layout = QVBoxLayout()
        self.fusion_previous_masks_checkbox = QCheckBox("Fusion previous masks")
        self.fusion_previous_masks_checkbox.setChecked(self.default_fusion_masks_state)
        self.fusion_previous_masks_checkbox.stateChanged.connect(self._changeFusionMaskState)
        self.remove_outliers_layout.addWidget(self.fusion_previous_masks_checkbox)
        
        self.keep_max_component_layout = QHBoxLayout()
        self.keep_max_component_checkbox = QCheckBox("Keep max connected component only")
        self.keep_max_component_checkbox.setChecked(self.keep_max_component_only)
        self.keep_max_component_checkbox.stateChanged.connect(self._updateKeepMaxComponentState)
        self.keep_max_component_layout.addWidget(self.keep_max_component_checkbox)
        self.remove_outliers_layout.addLayout(self.keep_max_component_layout)
        
        self.object_size_layout = QHBoxLayout()
        self.object_size_label = QLabel("Minimum object size:")
        self.object_size_entry = QLineEdit()
        self.object_size_entry.setValidator(QIntValidator())
        self.object_size_entry.setText(str(self.default_object_size))
        self.object_size_entry.editingFinished.connect(self._updateObjectSizeValue)
        self.object_size_layout.addWidget(self.object_size_label)
        self.object_size_layout.addWidget(self.object_size_entry)
        self.remove_outliers_layout.addLayout(self.object_size_layout)
        
        self.kernelSizeLayout = QHBoxLayout()
        self.closing_kernel_label = QLabel("Morphological operation kernel:")
        self.closing_kernel_label.setFont(QFont("Arial", 8, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        self.kernel_size_label = QLabel("Closing kernel size:")
        self.kernel_size_field = QLineEdit()
        self.kernel_size_field.setValidator(QIntValidator())
        self.kernel_size_field.setText(str(self.default_kernel_size))
        self.kernel_size_field.editingFinished.connect(self._updateKernelSizeValue)
        self.kernelSizeLayout.addWidget(self.kernel_size_label)
        self.kernelSizeLayout.addWidget(self.kernel_size_field)
        
        self.kernelShapeLayout = QHBoxLayout()
        self.kernel_shape_label = QLabel("Closing kernel shape:")
        self.kernel_shape_combobox = QComboBox()
        self.kernel_shape_combobox.addItems(list(self.kernel_shape_dict.keys()))
        self.kernel_shape_combobox.setCurrentIndex(self.default_kernel_shape_index)
        self.kernel_shape_combobox.currentIndexChanged.connect(self._updateKernelShape)
        self.kernelShapeLayout.addWidget(self.kernel_shape_label)
        self.kernelShapeLayout.addWidget(self.kernel_shape_combobox)
        
        self.remove_outliers_layout.addWidget(self.closing_kernel_label)
        self.remove_outliers_layout.addLayout(self.kernelSizeLayout)
        self.remove_outliers_layout.addLayout(self.kernelShapeLayout)
        layout.addLayout(self.remove_outliers_layout)
        
        # Boutons
        button_layout = QHBoxLayout()
        
        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self._applySettings)
        button_layout.addWidget(self.applyButton)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.close)
        button_layout.addWidget(self.cancelButton)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return panel
    
    def _createColorGroup(self, color_name, color_key):
        """Crée un groupe de contrôles pour une couleur"""
        group = QFrame()
        group.setFrameStyle(QFrame.Shape.Box if PYQT_VERSION == 6 else QFrame.Box)
        layout = QVBoxLayout(group)
        
        # Titre du groupe
        title = QLabel(f"{color_name} Channel")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        layout.addWidget(title)
        
        # Labels de valeurs
        value_layout = QHBoxLayout()
        min_label = QLabel("0")
        min_label.setFixedWidth(30)
        max_label = QLabel("255")
        max_label.setFixedWidth(30)
        value_layout.addWidget(min_label)
        value_layout.addStretch()
        value_layout.addWidget(max_label)
        layout.addLayout(value_layout)
        
        # Range slider
        range_slider = RangeSlider(Qt.Orientation.Horizontal if PYQT_VERSION == 6 else Qt.Horizontal)
        range_slider.setMinimum(0)
        range_slider.setMaximum(255)
        range_slider.setLow(0)
        range_slider.setHigh(255)
        range_slider.sliderMoved.connect(lambda low, high, key=color_key: self._updateColorRange(key, low, high))
        layout.addWidget(range_slider)
        
        # Labels des valeurs actuelles
        current_layout = QHBoxLayout()
        current_min = QLabel("0")
        current_min.setFixedWidth(30)
        current_max = QLabel("255")
        current_max.setFixedWidth(30)
        current_layout.addWidget(current_min)
        current_layout.addStretch()
        current_layout.addWidget(current_max)
        layout.addLayout(current_layout)
        
        # Checkbox invert
        invert_checkbox = QCheckBox("Invert values (keep values OUTSIDE range)")
        invert_checkbox.stateChanged.connect(lambda state, key=color_key: self._updateInvertState(key, state))
        layout.addWidget(invert_checkbox)
        
        # Stocker les références
        setattr(self, f"{color_key}_slider", range_slider)
        setattr(self, f"{color_key}_min_label", current_min)
        setattr(self, f"{color_key}_max_label", current_max)
        setattr(self, f"{color_key}_invert_checkbox", invert_checkbox)
        
        return group
    
    def _createPreviewPanel(self):
        """Crée le panel de prévisualisation"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel if PYQT_VERSION == 6 else QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        
        # Titre
        title = QLabel("Preview")
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold if PYQT_VERSION == 6 else QFont.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        layout.addWidget(title)
        
        # Widget de prévisualisation - maintenant extensible
        self.previewWidget = PreviewWidget()
        layout.addWidget(self.previewWidget)
        
        # Informations
        info_label = QLabel("Select an area in the main window to preview the threshold effect")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT_VERSION == 6 else Qt.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setFixedHeight(40)  # Hauteur fixe pour les infos
        layout.addWidget(info_label)
        
        return panel
    
    def _updateColorRange(self, color_key, low, high):
        """Met à jour les valeurs d'un canal de couleur"""
        if color_key == "red":
            self.red_lower_value = low
            self.red_upper_value = high
            self.red_min_label.setText(str(low))
            self.red_max_label.setText(str(high))
        elif color_key == "green":
            self.green_lower_value = low
            self.green_upper_value = high
            self.green_min_label.setText(str(low))
            self.green_max_label.setText(str(high))
        elif color_key == "blue":
            self.blue_lower_value = low
            self.blue_upper_value = high
            self.blue_min_label.setText(str(low))
            self.blue_max_label.setText(str(high))
        
        # Mettre à jour la prévisualisation
        self._updatePreview()
    
    def _updateInvertState(self, color_key, state):
        """Met à jour l'état d'inversion d'un canal"""
        inverted = bool(state)
        if color_key == "red":
            self.red_inverted_values = inverted
        elif color_key == "green":
            self.green_inverted_values = inverted
        elif color_key == "blue":
            self.blue_inverted_values = inverted
        
        # Mettre à jour la prévisualisation
        self._updatePreview()
    
    def _changeFusionMaskState(self):
        """Met à jour la valeur de la variable associée à l'état de la CheckBox de fusion des masques"""
        self.fusion_previous_masks = self.fusion_previous_masks_checkbox.isChecked()
        self._updatePreview()
    
    def _updateKeepMaxComponentState(self):
        self.keep_max_component_only = self.keep_max_component_checkbox.isChecked()
        # Mettre à jour la prévisualisation
        self._updatePreview()
    
    def _updateObjectSizeValue(self):
        """Met à jour la valeur de la taille minimum des objets"""
        try:
            object_size = int(self.object_size_entry.text())
            min_size = max(0, min(self.max_object_size_value, object_size))
            if min_size != object_size:
                self.object_size_entry.setText(str(self.min_object_size))
            
            self.min_object_size = min_size
            # Mettre à jour la prévisualisation
            self._updatePreview()
        except:
            print("Error: a strictly positive integer is expected for the value of the minimum size of objects.")
    
    def _updateKernelSizeValue(self):
        """Met à jour la taille du noyau pour l'opération morphologique de fermeture"""
        try:
            kernel_size = int(self.kernel_size_field.text())
            ksize = max(0, kernel_size)
            if ksize != kernel_size:
                self.kernel_size_field.setText(str(ksize))
            
            self.kernel_size = (ksize, ksize)
            # Mis à jour du noyau
            self._updateClosingKernel()
            # Mettre à jour la prévisualisation
            self._updatePreview()
        except:
            print("Error: a strictly positive integer is expected for the value of the closing kernel size.")
    
    def _updateKernelShape(self):
        """Met à jour la forme du noyau pour l'opération morphologique de fermeture"""
        current_index = self.kernel_shape_combobox.currentIndex()
        shape_name = list(self.kernel_shape_dict.keys())[current_index]
        self.kernel_shape = shape_name
        self.kernel_shape_value = self.kernel_shape_dict[shape_name]
        # Mis à jour du noyau
        self._updateClosingKernel()
        # Mettre à jour la prévisualisation
        self._updatePreview()
    
    def _updateClosingKernel(self):
        """Met à jour le noyau pour l'opération morphologique de fermeture"""
        if self.kernel_size == (0, 0):
            self.closing_kernel = None
        else:
            self.closing_kernel = cv2.getStructuringElement(self.kernel_shape_value, self.kernel_size)
    
    def _updatePreview(self):
        """Met à jour la prévisualisation"""
        self.previewWidget.updatePreview(
            (self.red_lower_value, self.red_upper_value),
            (self.green_lower_value, self.green_upper_value),
            (self.blue_lower_value, self.blue_upper_value),
            self.red_inverted_values,
            self.green_inverted_values,
            self.blue_inverted_values,
            closing_kernel=self.closing_kernel,
            keep_max_component_only=self.keep_max_component_only,
            object_size=self.min_object_size
        )
    
    def setFusionMaskState(self, check_state):
        """Active ou désactive la checkbox permettant l'activation ou désactivation de la fusion des masques"""
        try:
            self.fusion_previous_masks_checkbox.setCheckState(int(check_state)*2)
            self.fusion_previous_masks = check_state
        except:
            print("Error: \'check_state\' argument must be a boolean!")
    
    def setKernelShape(self, kernel_name):
        """Met à jour le type de noyau et la valeur de la combobox correspondante"""
        for i, kname in enumerate(self.kernel_shape_dict.keys()):
            if kname.lower() == kernel_name.lower():
                self.kernel_shape = kernel_name
                self.kernel_shape_value = self.kernel_shape_dict[kernel_name]
                self.kernel_shape_combobox.setCurrentIndex(i)
                break
    
    def setImageData(self, image_array, selection_coords=None):
        """Définit l'image et la sélection pour la prévisualisation"""
        self.max_object_size_value = image_array.shape[0]*image_array.shape[1]
        self.previewWidget.setOriginalImage(image_array)
        self.previewWidget.setSelectionCoords(selection_coords)
        self._updatePreview()
    
    def _applySettings(self):
        """Applique les réglages et ferme la fenêtre"""
        print("Applied settings:")
        print(f"Red: [{self.red_lower_value}-{self.red_upper_value}] Inverted: {self.red_inverted_values}")
        print(f"Green: [{self.green_lower_value}-{self.green_upper_value}] Inverted: {self.green_inverted_values}")
        print(f"Blue: [{self.blue_lower_value}-{self.blue_upper_value}] Inverted: {self.blue_inverted_values}")
        
        # Emet un signal vers l'application principale
        if self.parent_app:
            self.parent_app.setColorThresholdValues(red_interval=[self.red_lower_value, self.red_upper_value], 
                                                    blue_interval=[self.blue_lower_value, self.blue_upper_value],
                                                    green_interval=[self.green_lower_value, self.green_upper_value],
                                                    invert_red=self.red_inverted_values,
                                                    invert_blue=self.blue_inverted_values,
                                                    invert_green=self.green_inverted_values)
            self.parent_app.exportSegmentedImages()
        self.close()
    
    def getThresholdParameters(self):
        """Retourne les paramètres de seuillage actuels"""
        return {
            'red_range': (self.red_lower_value, self.red_upper_value),
            'green_range': (self.green_lower_value, self.green_upper_value),
            'blue_range': (self.blue_lower_value, self.blue_upper_value),
            'red_invert': self.red_inverted_values,
            'green_invert': self.green_inverted_values,
            'blue_invert': self.blue_inverted_values
        }


# Fonction d'exemple pour tester la fenêtre
def test_threshold_window():
    app = QApplication([])
    
    # Créer une image de test
    test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    test_image[50:150, 100:200] = [255, 0, 0]  # Zone rouge
    
    window = EnhancedOptionsWindow("RGB Threshold Test", "icons")
    window.setImageData(test_image)
    window.show()
    
    if PYQT_VERSION == 6:
        app.exec()
    else:
        app.exec_()


if __name__ == "__main__":
    test_threshold_window()
