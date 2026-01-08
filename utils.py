import numpy as np

from contextlib import contextmanager

PYQT_AVAILABLE = False
PYQT_VERSION = None

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
except:
    try:
        from PyQt5.QtCore import Qt
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