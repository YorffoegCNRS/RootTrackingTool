import numpy as np

from contextlib import contextmanager

PYQT_AVAILABLE = False
PYQT_VERSION = None

try:
    from PyQt6.QtCore import Qt, QObject
    from PyQt6.QtWidgets import QApplication
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
except:
    try:
        from PyQt5.QtCore import Qt, QObject
        from PyQt5.QtWidgets import QApplication
        PYQT_AVAILABLE = True
        PYQT_VERSION = 5
    except:
        print("This script requires PyQt5 or PyQt6 to run. Neither of these versions was found!")



def _wait_cursor():
    """Qt wait cursor compatible PyQt5/PyQt6."""
    if PYQT_VERSION == 6:
        try:
            return Qt.CursorShape.WaitCursor
        except Exception:
            return Qt.WaitCursor
    return Qt.WaitCursor


class WheelToScrollAreaFilter(QObject):
    """Empêche la roulette de modifier les QSpinBox/QDoubleSpinBox/QComboBox tant qu'ils n'ont pas le focus.
    À la place, on fait défiler la QScrollArea (panneau de gauche).
    
    Objectif : éviter les changements involontaires de paramètres lorsque l'utilisateur veut seulement scroller.
    """
    
    def __init__(self, scroll_area, parent=None):
        super().__init__(parent)
        self._scroll_area = scroll_area
    
    def eventFilter(self, obj, event):
        try:
            etype = event.type()
            wheel_type = QEvent.Type.Wheel if hasattr(QEvent, "Type") else QEvent.Wheel
            if etype == wheel_type:
                # Si le widget n'a pas le focus, on redirige la roulette vers la scroll area
                if hasattr(obj, "hasFocus") and not obj.hasFocus():
                    sa = self._scroll_area
                    try:
                        bar = sa.verticalScrollBar()
                    except Exception:
                        return True  # on bloque quand même
                    # Delta standard (angleDelta) : 120 = un cran
                    dy = 0
                    try:
                        dy = int(event.angleDelta().y())
                    except Exception:
                        try:
                            dy = int(event.delta())
                        except Exception:
                            dy = 0
                    if dy != 0:
                        step = bar.singleStep()
                        # facteur empirique pour un scroll fluide, sans être trop rapide
                        bar.setValue(bar.value() - int((dy / 120.0) * step * 3))
                    return True  # on bloque la modification de valeur
        except Exception:
            # En cas de doute, on n'empêche pas l'event
            return False
        return False


@contextmanager
def ui_busy(window, disable_widgets=None, set_wait_cursor=True):
    """
    window: QMainWindow / QWidget principal
    disable_widgets: list[QWidget] à désactiver (optionnel)
    """
    if disable_widgets is None:
        disable_widgets = []
    
    old_cursor = None
    try:
        # 1) Désactiver widgets
        for w in disable_widgets:
            if w is not None:
                w.setEnabled(False)
        
        # 2) Curseur busy
        if set_wait_cursor:
            QApplication.setOverrideCursor(_wait_cursor())
        
        yield
    
    finally:
        # Toujours exécuté, même en cas d'erreur
        if set_wait_cursor:
            QApplication.restoreOverrideCursor()
        
        for w in disable_widgets:
            if w is not None:
                w.setEnabled(True)



def validate_rgb_mask(mask, name="mask"):
    if mask is None:
        raise ValueError(f"{name} is None")
    
    if mask.ndim != 3 or mask.shape[2] != 3:
        raise ValueError(f"{name} must be HxWx3, got shape={mask.shape}")
    
    h, w, c = mask.shape
    if h < 2 or w < 2:
        raise ValueError(f"{name} too small: shape={mask.shape}")


def validate_same_shape(a, b, name_a="a", name_b="b"):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {name_a}={a.shape} vs {name_b}={b.shape}")