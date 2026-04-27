# compat.py


# Scientific packages
try:
    import numpy as np
    NUMPY_VERSION = np.__version__
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available")
    NUMPY_VERSION = None
    NUMPY_AVAILABLE = False


# PyQt imports - detect version
PYQT_AVAILABLE = False
PYQT_VERSION = None
QT_API = None
QT_VERSION = None
BINDING_VERSION = None


# Handle PyQt/PySide versions import
try:
    from PyQt6 import QtCore, QtWidgets, QtGui
    from PyQt6.QtCore import (
        Qt, QPoint, QPointF, QRect, QRectF, QTimer, QSize, QLocale, QThread, QEvent, QObject
    )
    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, pyqtProperty as Property
    
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QDialog, QLabel, QMessageBox,
        QVBoxLayout, QHBoxLayout, QPushButton, QToolButton, QFrame, 
        QFileDialog, QInputDialog, QTextEdit, QGridLayout, QLineEdit,
        QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QCheckBox,
        QRadioButton, QMenu, QMenuBar, QSizePolicy, QProgressDialog, 
        QTabWidget, QListWidget, QListWidgetItem, QSplitter, QProgressBar, 
        QScrollArea, QGroupBox, QSpinBox, QButtonGroup, QFormLayout, 
        QTableWidget, QTableWidgetItem
    )
    
    from PyQt6.QtGui import (
        QPixmap, QImage, QColor, QFont, QFontMetrics, QIcon, QBitmap, QPainter,
        QPen, QBrush, QRegion, QIntValidator, QDoubleValidator, QKeySequence, QAction
    )
    
    PYQT_AVAILABLE = True
    PYQT_VERSION = 6
    QT_API = "PyQt6"
    QT_VERSION = getattr(QtCore, "QT_VERSION_STR", None)
    QT_HAS_POSITION = True
    BINDING_VERSION = getattr(QtCore, "PYQT_VERSION_STR", None)
    
    ### Compatibility of various PyQt/PySide properties
    class QtCompat:
        AlignCenter = Qt.AlignmentFlag.AlignCenter
        
        AlignLeft = Qt.AlignmentFlag.AlignLeft
        AlignRight = Qt.AlignmentFlag.AlignRight
        AlignHCenter = Qt.AlignmentFlag.AlignHCenter
        AlignJustify = Qt.AlignmentFlag.AlignJustify
        
        AlignTop = Qt.AlignmentFlag.AlignTop
        AlignBottom = Qt.AlignmentFlag.AlignBottom
        AlignVCenter = Qt.AlignmentFlag.AlignVCenter
        AlignBaseline = Qt.AlignmentFlag.AlignBaseline
        
        AbsoluteSize = Qt.SizeMode.AbsoluteSize
        RelativeSize = Qt.SizeMode.RelativeSize
        
        AscendingOrder = Qt.SortOrder.AscendingOrder
        DescendingOrder = Qt.SortOrder.DescendingOrder
        
        IgnoreAspectRatio = Qt.AspectRatioMode.IgnoreAspectRatio
        KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
        KeepAspectRatioByExpanding = Qt.AspectRatioMode.KeepAspectRatioByExpanding
        
        FastTransformation = Qt.TransformationMode.FastTransformation
        SmoothTransformation = Qt.TransformationMode.SmoothTransformation
        
        Horizontal = Qt.Orientation.Horizontal
        Vertical = Qt.Orientation.Vertical
    
    ### BrushStyle compatibility
    class BrushStyle:
        NoBrush = Qt.BrushStyle.NoBrush
        SolidPattern = Qt.BrushStyle.SolidPattern
        Dense1Pattern = Qt.BrushStyle.Dense1Pattern
        Dense2Pattern = Qt.BrushStyle.Dense2Pattern
        Dense3Pattern = Qt.BrushStyle.Dense3Pattern
        Dense4Pattern = Qt.BrushStyle.Dense4Pattern
        Dense5Pattern = Qt.BrushStyle.Dense5Pattern
        Dense6Pattern = Qt.BrushStyle.Dense6Pattern
        Dense7Pattern = Qt.BrushStyle.Dense7Pattern
        
        HorPattern = Qt.BrushStyle.HorPattern
        VerPattern = Qt.BrushStyle.VerPattern
        CrossPattern = Qt.BrushStyle.CrossPattern
        BDiagPattern = Qt.BrushStyle.BDiagPattern
        FDiagPattern = Qt.BrushStyle.FDiagPattern
        DiagCrossPattern = Qt.BrushStyle.DiagCrossPattern
        
        LinearGradientPattern = Qt.BrushStyle.LinearGradientPattern
        ConicalGradientPattern = Qt.BrushStyle.ConicalGradientPattern
        RadialGradientPattern = Qt.BrushStyle.RadialGradientPattern
        TexturePattern = Qt.BrushStyle.TexturePattern
    
    ### CheckState compatibility
    class CheckState:
        Unchecked = Qt.CheckState.Unchecked
        PartiallyChecked = Qt.CheckState.PartiallyChecked
        Checked = Qt.CheckState.Checked
    
    ### CursorShape compatibility
    class CursorShape:
        ArrowCursor = Qt.CursorShape.ArrowCursor
        UpArrowCursor = Qt.CursorShape.UpArrowCursor
        CrossCursor = Qt.CursorShape.CrossCursor
        WaitCursor = Qt.CursorShape.WaitCursor
        
        IBeamCursor = Qt.CursorShape.IBeamCursor
        SizeVerCursor = Qt.CursorShape.SizeVerCursor
        SizeHorCursor = Qt.CursorShape.SizeHorCursor
        SizeBDiagCursor = Qt.CursorShape.SizeBDiagCursor
        SizeFDiagCursor = Qt.CursorShape.SizeFDiagCursor
        SizeAllCursor = Qt.CursorShape.SizeAllCursor
        
        BlankCursor = Qt.CursorShape.BlankCursor
        SplitVCursor = Qt.CursorShape.SplitVCursor
        SplitHCursor = Qt.CursorShape.SplitHCursor
        
        PointingHandCursor = Qt.CursorShape.PointingHandCursor
        ForbiddenCursor = Qt.CursorShape.ForbiddenCursor
        OpenHandCursor = Qt.CursorShape.OpenHandCursor
        ClosedHandCursor = Qt.CursorShape.ClosedHandCursor
        WhatsThisCursor = Qt.CursorShape.WhatsThisCursor
        BusyCursor = Qt.CursorShape.BusyCursor
        
        DragMoveCursor = Qt.CursorShape.DragMoveCursor
        DragCopyCursor = Qt.CursorShape.DragCopyCursor
        DragLinkCursor = Qt.CursorShape.DragLinkCursor
        BitmapCursor = Qt.CursorShape.BitmapCursor
    
    ### QEvent compatibility
    class EventType:
        Clipboard = QEvent.Type.Clipboard
        Close = QEvent.Type.Close
        CursorChange = QEvent.Type.CursorChange
        Enter = QEvent.Type.Enter
        FontChange = QEvent.Type.FontChange
        Leave = QEvent.Type.Leave
        
        Hide = QEvent.Type.Hide
        HideToParent = QEvent.Type.HideToParent
        
        HoverEnter = QEvent.Type.HoverEnter
        HoverLeave = QEvent.Type.HoverLeave
        HoverMove = QEvent.Type.HoverMove
        
        KeyPress = QEvent.Type.KeyPress
        KeyRelease = QEvent.Type.KeyRelease
        
        MouseButtonPress = QEvent.Type.MouseButtonPress
        MouseButtonRelease = QEvent.Type.MouseButtonRelease
        MouseMove = QEvent.Type.MouseMove
        MouseTrackingChange = QEvent.Type.MouseTrackingChange
        
        Paint = QEvent.Type.Paint
        PaletteChange = QEvent.Type.PaletteChange
        
        Resize = QEvent.Type.Resize
        Scroll = QEvent.Type.Scroll
        Shortcut = QEvent.Type.Shortcut
        
        Show = QEvent.Type.Show
        ShowToParent = QEvent.Type.ShowToParent
        StyleChange = QEvent.Type.StyleChange
        
        Timer = QEvent.Type.Timer
        ToolBarChange = QEvent.Type.ToolBarChange
        ToolTip = QEvent.Type.ToolTip
        ToolTipChange = QEvent.Type.ToolTipChange
        
        UpdateLater = QEvent.Type.UpdateLater
        UpdateRequest = QEvent.Type.UpdateRequest
        
        Wheel = QEvent.Type.Wheel
    
    ### Focus Policy compatibility
    class FocusPolicy:
        TabFocus = Qt.FocusPolicy.TabFocus
        ClickFocus = Qt.FocusPolicy.ClickFocus
        StrongFocus = Qt.FocusPolicy.StrongFocus
        WheelFocus = Qt.FocusPolicy.WheelFocus
        NoFocus = Qt.FocusPolicy.NoFocus
    
    ### QFont Compatibility
    class FontWeight:
        ExtraBold = QFont.Weight.ExtraBold
        Black = QFont.Weight.Black
        Bold = QFont.Weight.Bold
        DemiBold = QFont.Weight.DemiBold
        Medium = QFont.Weight.Medium
        Normal = QFont.Weight.Normal
        Light = QFont.Weight.Light
        ExtraLight = QFont.Weight.ExtraLight
        Thin = QFont.Weight.Thin
    
    class FontStyle:
        Normal = QFont.Style.StyleNormal
        Italic = QFont.Style.StyleItalic
        Oblique = QFont.Style.StyleOblique
    
    ### QFrame compatibility
    class FrameShadow:
        Plain = QFrame.Shadow.Plain
        Raised = QFrame.Shadow.Raised
        Sunken = QFrame.Shadow.Sunken
    
    class FrameShape:
        NoFrame = QFrame.Shape.NoFrame
        Box = QFrame.Shape.Box
        Panel = QFrame.Shape.Panel
        StyledPanel = QFrame.Shape.StyledPanel
        HLine = QFrame.Shape.HLine
        VLine = QFrame.Shape.VLine
        WinPanel = QFrame.Shape.WinPanel
    
    ### GlobalColor compatibility
    class GlobalColor:
        White = Qt.GlobalColor.white
        Black = Qt.GlobalColor.black
        Red = Qt.GlobalColor.red
        DarkRed = Qt.GlobalColor.darkRed
        Green = Qt.GlobalColor.green
        DarkGreen = Qt.GlobalColor.darkGreen
        Blue = Qt.GlobalColor.blue
        DarkBlue = Qt.GlobalColor.darkBlue
        Cyan = Qt.GlobalColor.cyan
        DarkCyan = Qt.GlobalColor.darkCyan
        Magenta = Qt.GlobalColor.magenta
        DarkMagenta = Qt.GlobalColor.darkMagenta
        Yellow = Qt.GlobalColor.yellow
        DarkYellow = Qt.GlobalColor.darkYellow
        Gray = Qt.GlobalColor.gray
        DarkGray = Qt.GlobalColor.darkGray
        LightGray = Qt.GlobalColor.lightGray
        Transparent = Qt.GlobalColor.transparent
        Color0 = Qt.GlobalColor.color0
        Color1 = Qt.GlobalColor.color1
    
    ### QImage compatibility
    class ImageFormat:
        Format_RGB32 = QImage.Format.Format_RGB32
        Format_ARGB32 = QImage.Format.Format_ARGB32
                   
        Format_RGB888 = QImage.Format.Format_RGB888
        Format_RGBA8888 = QImage.Format_RGBA8888
        Format_ARGB8565_Premultiplied = QImage.Format.Format_ARGB8565_Premultiplied
        
        Format_RGB666 = QImage.Format.Format_RGB666
        Format_ARGB6666_Premultiplied = QImage.Format.Format_ARGB6666_Premultiplied
        
        Format_RGB555 = QImage.Format.Format_RGB555
        Format_ARGB8555_Premultiplied = QImage.Format.Format_ARGB8555_Premultiplied
        
        Format_RGB16 = QImage.Format.Format_RGB16
        
        Format_RGB444 = QImage.Format.Format_RGB444
        Format_ARGB4444_Premultiplied = QImage.Format.Format_ARGB4444_Premultiplied
        
        Format_Indexed8 = QImage.Format.Format_Indexed8
        
        Format_Invalid = QImage.Format.Format_Invalid
        Format_Mono = QImage.Format.Format_Mono
        Format_MonoLSB = QImage.Format.Format_MonoLSB
    
    ### ItemFlag compatibility
    class ItemFlag:
        NoItemFlags = Qt.ItemFlag.NoItemFlags
        ItemIsSelectable = Qt.ItemFlag.ItemIsSelectable
        ItemIsEditable = Qt.ItemFlag.ItemIsEditable
        ItemIsDragEnabled = Qt.ItemFlag.ItemIsDragEnabled
        ItemIsDropEnabled = Qt.ItemFlag.ItemIsDropEnabled
        ItemIsUserCheckable = Qt.ItemFlag.ItemIsUserCheckable
        ItemIsEnabled = Qt.ItemFlag.ItemIsEnabled
        ItemIsAutoTristate = Qt.ItemFlag.ItemIsAutoTristate
        ItemNeverHasChildren = Qt.ItemFlag.ItemNeverHasChildren
        ItemIsUserTristate = Qt.ItemFlag.ItemIsUserTristate
    
    ### MouseButton compatibility
    class MouseButton:
        NoButton = Qt.MouseButton.NoButton
        AllButtons = Qt.MouseButton.AllButtons
        
        LeftButton = Qt.MouseButton.LeftButton
        RightButton = Qt.MouseButton.RightButton
        MiddleButton = Qt.MouseButton.MiddleButton
        BackButton = Qt.MouseButton.BackButton
        
        ForwardButton = Qt.MouseButton.ForwardButton
        TaskButton = Qt.MouseButton.TaskButton
        XButton1 = Qt.MouseButton.XButton1
        XButton2 = Qt.MouseButton.XButton2
        
        ExtraButton1 = Qt.MouseButton.ExtraButton1
        ExtraButton2 = Qt.MouseButton.ExtraButton2
        ExtraButton3 = Qt.MouseButton.ExtraButton3
        ExtraButton4 = Qt.MouseButton.ExtraButton4
        ExtraButton5 = Qt.MouseButton.ExtraButton5
        ExtraButton6 = Qt.MouseButton.ExtraButton6
        ExtraButton7 = Qt.MouseButton.ExtraButton7
        ExtraButton8 = Qt.MouseButton.ExtraButton8
        ExtraButton9 = Qt.MouseButton.ExtraButton9
        ExtraButton10 = Qt.MouseButton.ExtraButton10
        ExtraButton11 = Qt.MouseButton.ExtraButton11
        ExtraButton12 = Qt.MouseButton.ExtraButton12
        ExtraButton13 = Qt.MouseButton.ExtraButton13
        ExtraButton14 = Qt.MouseButton.ExtraButton14
        ExtraButton15 = Qt.MouseButton.ExtraButton15
        ExtraButton16 = Qt.MouseButton.ExtraButton16
        ExtraButton17 = Qt.MouseButton.ExtraButton17
        ExtraButton18 = Qt.MouseButton.ExtraButton18
        ExtraButton19 = Qt.MouseButton.ExtraButton19
        ExtraButton20 = Qt.MouseButton.ExtraButton20
        ExtraButton21 = Qt.MouseButton.ExtraButton21
        ExtraButton22 = Qt.MouseButton.ExtraButton22
        ExtraButton23 = Qt.MouseButton.ExtraButton23
        ExtraButton24 = Qt.MouseButton.ExtraButton24
    
    ### PenStyle compatibility
    class PenStyle:
        NoPen = Qt.PenStyle.NoPen
        SolidLine = Qt.PenStyle.SolidLine
        DashLine = Qt.PenStyle.DashLine
        DotLine = Qt.PenStyle.DotLine
        DashDotLine = Qt.PenStyle.DashDotLine
        DashDotDotLine = Qt.PenStyle.DashDotDotLine
        CustomDashLine = Qt.PenStyle.CustomDashLine
    
    ### RenderHint compatibility
    class RenderHint:
        Antialiasing = QPainter.RenderHint.Antialiasing
        TextAntialiasing = QPainter.RenderHint.TextAntialiasing
        SmoothPixmapTransform = QPainter.RenderHint.SmoothPixmapTransform
    
    ### ScrollBarPolicy compatibility
    class ScrollBarPolicy:
        ScrollBarAsNeeded = Qt.ScrollBarPolicy.ScrollBarAsNeeded
        ScrollBarAlwaysOff = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        ScrollBarAlwaysOn = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    
    ### QSizePolicy compatibility
    class SizePolicy:
        Fixed = QSizePolicy.Policy.Fixed
        Minimum = QSizePolicy.Policy.Minimum
        Maximum = QSizePolicy.Policy.Maximum
        Preferred = QSizePolicy.Policy.Preferred
        Expanding = QSizePolicy.Policy.Expanding
        MinimumExpanding = QSizePolicy.Policy.MinimumExpanding
        Ignored = QSizePolicy.Policy.Ignored
except ImportError:
    try:
        from PyQt5 import QtCore, QtWidgets, QtGui
        from PyQt5.QtCore import (
            Qt, QPoint, QPointF, QRect, QRectF, QTimer, QSize, QLocale, QThread, QEvent, QObject
        )
        from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot, pyqtProperty as Property
        
        from PyQt5.QtWidgets import (
            QApplication, QWidget, QMainWindow, QDialog, QLabel, QMessageBox,
            QVBoxLayout, QHBoxLayout, QPushButton, QToolButton, QFrame, 
            QFileDialog, QInputDialog, QTextEdit, QGridLayout, QLineEdit,
            QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QCheckBox,
            QRadioButton, QMenu, QMenuBar, QSizePolicy, QProgressDialog, 
            QTabWidget, QListWidget, QListWidgetItem, QSplitter, QProgressBar, 
            QScrollArea, QAction, QGroupBox, QSpinBox, QButtonGroup, QFormLayout,
            QTableWidget, QTableWidgetItem
        )
        
        from PyQt5.QtGui import (
            QPixmap, QImage, QColor, QFont, QFontMetrics, QIcon, QBitmap, QPainter,
            QPen, QBrush, QRegion, QIntValidator, QDoubleValidator, QKeySequence
        )
        
        
        PYQT_AVAILABLE = True
        PYQT_VERSION = 5
        QT_API = "PyQt5"
        QT_VERSION = getattr(QtCore, "QT_VERSION_STR", None)
        QT_HAS_POSITION = False
        BINDING_VERSION = getattr(QtCore, "PYQT_VERSION_STR", None)
        
        ### Compatibility of various PyQt/PySide properties
        class QtCompat:
            AlignCenter = Qt.AlignCenter
            
            AlignLeft = Qt.AlignLeft
            AlignRight = Qt.AlignRight
            AlignHCenter = Qt.AlignHCenter
            AlignJustify = Qt.AlignJustify
            
            AlignTop = Qt.AlignTop
            AlignBottom = Qt.AlignBottom
            AlignVCenter = Qt.AlignVCenter
            AlignBaseline = Qt.AlignBaseline
            
            AbsoluteSize = Qt.AbsoluteSize
            RelativeSize = Qt.RelativeSize
            
            AscendingOrder = Qt.AscendingOrder
            DescendingOrder = Qt.DescendingOrder
            
            IgnoreAspectRatio = Qt.IgnoreAspectRatio
            KeepAspectRatio = Qt.KeepAspectRatio
            KeepAspectRatioByExpanding = Qt.KeepAspectRatioByExpanding
            
            FastTransformation = Qt.FastTransformation
            SmoothTransformation = Qt.SmoothTransformation
            
            Horizontal = Qt.Horizontal
            Vertical = Qt.Vertical
        
        ### BrushStyle compatibility
        class BrushStyle:
            NoBrush = Qt.NoBrush
            SolidPattern = Qt.SolidPattern
            Dense1Pattern = Qt.Dense1Pattern
            Dense2Pattern = Qt.Dense2Pattern
            Dense3Pattern = Qt.Dense3Pattern
            Dense4Pattern = Qt.Dense4Pattern
            Dense5Pattern = Qt.Dense5Pattern
            Dense6Pattern = Qt.Dense6Pattern
            Dense7Pattern = Qt.Dense7Pattern
            
            HorPattern = Qt.HorPattern
            VerPattern = Qt.VerPattern
            CrossPattern = Qt.CrossPattern
            BDiagPattern = Qt.BDiagPattern
            FDiagPattern = Qt.FDiagPattern
            DiagCrossPattern = Qt.DiagCrossPattern
            
            LinearGradientPattern = Qt.LinearGradientPattern
            ConicalGradientPattern = Qt.ConicalGradientPattern
            RadialGradientPattern = Qt.RadialGradientPattern
            TexturePattern = Qt.TexturePattern
        
        ### CheckState compatibility
        class CheckState:
            Unchecked = Qt.Unchecked
            PartiallyChecked = Qt.PartiallyChecked
            Checked = Qt.Checked
        
        ### CursorShape compatibility
        class CursorShape:
            ArrowCursor = Qt.ArrowCursor
            UpArrowCursor = Qt.UpArrowCursor
            CrossCursor = Qt.CrossCursor
            WaitCursor = Qt.WaitCursor
            
            IBeamCursor = Qt.IBeamCursor
            SizeVerCursor = Qt.SizeVerCursor
            SizeHorCursor = Qt.SizeHorCursor
            SizeBDiagCursor = Qt.SizeBDiagCursor
            SizeFDiagCursor = Qt.SizeFDiagCursor
            SizeAllCursor = Qt.SizeAllCursor
            
            BlankCursor = Qt.BlankCursor
            SplitVCursor = Qt.SplitVCursor
            SplitHCursor = Qt.SplitHCursor
            
            PointingHandCursor = Qt.PointingHandCursor
            ForbiddenCursor = Qt.ForbiddenCursor
            OpenHandCursor = Qt.OpenHandCursor
            ClosedHandCursor = Qt.ClosedHandCursor
            WhatsThisCursor = Qt.WhatsThisCursor
            BusyCursor = Qt.BusyCursor
            
            DragMoveCursor = Qt.DragMoveCursor
            DragCopyCursor = Qt.DragCopyCursor
            DragLinkCursor = Qt.DragLinkCursor
            BitmapCursor = Qt.BitmapCursor
        
        ### QEvent compatibility
        class EventType:
            Clipboard = QEvent.Clipboard
            Close = QEvent.Close
            CursorChange = QEvent.CursorChange
            Enter = QEvent.Enter
            FontChange = QEvent.FontChange
            Leave = QEvent.Leave
            
            Hide = QEvent.Hide
            HideToParent = QEvent.HideToParent
            
            HoverEnter = QEvent.HoverEnter
            HoverLeave = QEvent.HoverLeave
            HoverMove = QEvent.HoverMove
            
            KeyPress = QEvent.KeyPress
            KeyRelease = QEvent.KeyRelease
            
            MouseButtonPress = QEvent.MouseButtonPress
            MouseButtonRelease = QEvent.MouseButtonRelease
            MouseMove = QEvent.MouseMove
            MouseTrackingChange = QEvent.MouseTrackingChange
            
            Paint = QEvent.Paint
            PaletteChange = QEvent.PaletteChange
            
            Resize = QEvent.Resize
            Scroll = QEvent.Scroll
            Shortcut = QEvent.Shortcut
            
            Show = QEvent.Show
            ShowToParent = QEvent.ShowToParent
            StyleChange = QEvent.StyleChange
            
            Timer = QEvent.Timer
            ToolBarChange = QEvent.ToolBarChange
            ToolTip = QEvent.ToolTip
            ToolTipChange = QEvent.ToolTipChange
            
            UpdateLater = QEvent.UpdateLater
            UpdateRequest = QEvent.UpdateRequest
            
            Wheel = QEvent.Wheel
        
        ### Focus Policy compatibility
        class FocusPolicy:
            TabFocus = Qt.TabFocus
            ClickFocus = Qt.ClickFocus
            StrongFocus = Qt.StrongFocus
            WheelFocus = Qt.WheelFocus
            NoFocus = Qt.NoFocus
        
        ### QFont Compatibility
        class FontWeight:
            ExtraBold = QFont.ExtraBold
            Black = QFont.Black
            Bold = QFont.Bold
            DemiBold = QFont.DemiBold
            Medium = QFont.Medium
            Normal = QFont.Normal
            Light = QFont.Light
            ExtraLight = QFont.ExtraLight
            Thin = QFont.Thin
        
        class FontStyle:
            Normal = QFont.StyleNormal
            Italic = QFont.StyleItalic
            Oblique = QFont.StyleOblique
        
        ### QFrame compatibility
        class FrameShadow:
            Plain = QFrame.Plain
            Raised = QFrame.Raised
            Sunken = QFrame.Sunken
        
        class FrameShape:
            NoFrame = QFrame.NoFrame
            Box = QFrame.Box
            Panel = QFrame.Panel
            StyledPanel = QFrame.StyledPanel
            HLine = QFrame.HLine
            VLine = QFrame.VLine
            WinPanel = QFrame.WinPanel
        
        ### GlobalColor compatibility
        class GlobalColor:
            White = Qt.white
            Black = Qt.black
            Red = Qt.red
            DarkRed = Qt.darkRed
            Green = Qt.green
            DarkGreen = Qt.darkGreen
            Blue = Qt.blue
            DarkBlue = Qt.darkBlue
            Cyan = Qt.cyan
            DarkCyan = Qt.darkCyan
            Magenta = Qt.magenta
            DarkMagenta = Qt.darkMagenta
            Yellow = Qt.yellow
            DarkYellow = Qt.darkYellow
            Gray = Qt.gray
            DarkGray = Qt.darkGray
            LightGray = Qt.lightGray
            Transparent = Qt.transparent
            Color0 = Qt.color0
            Color1 = Qt.color1
        
        ### QImage compatibility
        class ImageFormat:
            Format_RGB32 = QImage.Format_RGB32
            Format_ARGB32 = QImage.Format_ARGB32
                       
            Format_RGB888 = QImage.Format_RGB888
            Format_RGBA8888 = QImage.Format_RGBA8888
            Format_ARGB8565_Premultiplied = QImage.Format_ARGB8565_Premultiplied
            
            Format_RGB666 = QImage.Format_RGB666
            Format_ARGB6666_Premultiplied = QImage.Format_ARGB6666_Premultiplied
            
            Format_RGB555 = QImage.Format_RGB555
            Format_ARGB8555_Premultiplied = QImage.Format_ARGB8555_Premultiplied
            
            Format_RGB16 = QImage.Format_RGB16
            
            Format_RGB444 = QImage.Format_RGB444
            Format_ARGB4444_Premultiplied = QImage.Format_ARGB4444_Premultiplied
            
            Format_Indexed8 = QImage.Format_Indexed8
            
            Format_Invalid = QImage.Format_Invalid
            Format_Mono = QImage.Format_Mono
            Format_MonoLSB = QImage.Format_MonoLSB
        
        ### ItemFlag compatibility
        class ItemFlag:
            NoItemFlags = Qt.NoItemFlags
            ItemIsSelectable = Qt.ItemIsSelectable
            ItemIsEditable = Qt.ItemIsEditable
            ItemIsDragEnabled = Qt.ItemIsDragEnabled
            ItemIsDropEnabled = Qt.ItemIsDropEnabled
            ItemIsUserCheckable = Qt.ItemIsUserCheckable
            ItemIsEnabled = Qt.ItemIsEnabled
            ItemIsAutoTristate = Qt.ItemIsAutoTristate
            ItemNeverHasChildren = Qt.ItemNeverHasChildren
            ItemIsUserTristate = Qt.ItemIsUserTristate
        
        ### MouseButton compatibility
        class MouseButton:
            NoButton = Qt.NoButton
            AllButtons = Qt.AllButtons
            
            LeftButton = Qt.LeftButton
            RightButton = Qt.RightButton
            MiddleButton = Qt.MiddleButton
            BackButton = Qt.BackButton
            
            ForwardButton = Qt.ForwardButton
            TaskButton = Qt.TaskButton
            XButton1 = Qt.XButton1
            XButton2 = Qt.XButton2
            
            ExtraButton1 = Qt.ExtraButton1
            ExtraButton2 = Qt.ExtraButton2
            ExtraButton3 = Qt.ExtraButton3
            ExtraButton4 = Qt.ExtraButton4
            ExtraButton5 = Qt.ExtraButton5
            ExtraButton6 = Qt.ExtraButton6
            ExtraButton7 = Qt.ExtraButton7
            ExtraButton8 = Qt.ExtraButton8
            ExtraButton9 = Qt.ExtraButton9
            ExtraButton10 = Qt.ExtraButton10
            ExtraButton11 = Qt.ExtraButton11
            ExtraButton12 = Qt.ExtraButton12
            ExtraButton13 = Qt.ExtraButton13
            ExtraButton14 = Qt.ExtraButton14
            ExtraButton15 = Qt.ExtraButton15
            ExtraButton16 = Qt.ExtraButton16
            ExtraButton17 = Qt.ExtraButton17
            ExtraButton18 = Qt.ExtraButton18
            ExtraButton19 = Qt.ExtraButton19
            ExtraButton20 = Qt.ExtraButton20
            ExtraButton21 = Qt.ExtraButton21
            ExtraButton22 = Qt.ExtraButton22
            ExtraButton23 = Qt.ExtraButton23
            ExtraButton24 = Qt.ExtraButton24
        
        ### PenStyle compatibility
        class PenStyle:
            NoPen = Qt.PenStyle.NoPen
            SolidLine = Qt.PenStyle.SolidLine
            DashLine = Qt.PenStyle.DashLine
            DotLine = Qt.PenStyle.DotLine
            DashDotLine = Qt.PenStyle.DashDotLine
            DashDotDotLine = Qt.PenStyle.DashDotDotLine
            CustomDashLine = Qt.PenStyle.CustomDashLine
        
        ### RenderHint compatibility
        class RenderHint:
            Antialiasing = QPainter.Antialiasing
            TextAntialiasing = QPainter.TextAntialiasing
            SmoothPixmapTransform = QPainter.SmoothPixmapTransform
        
        ### ScrollBarPolicy compatibility
        class ScrollBarPolicy:
            ScrollBarAsNeeded = Qt.ScrollBarAsNeeded
            ScrollBarAlwaysOff = Qt.ScrollBarAlwaysOff
            ScrollBarAlwaysOn = Qt.ScrollBarAlwaysOn
        
        ### QSizePolicy compatibility
        class SizePolicy:
            Fixed = QSizePolicy.Fixed
            Minimum = QSizePolicy.Minimum
            Maximum = QSizePolicy.Maximum
            Preferred = QSizePolicy.Preferred
            Expanding = QSizePolicy.Expanding
            MinimumExpanding = QSizePolicy.MinimumExpanding
            Ignored = QSizePolicy.Ignored
    except ImportError:
        try:
            from PySide6 import QtCore, QtWidgets, QtGui
            from PySide6.QtCore import (
                Qt, QPoint, QPointF, QRect, QRectF, QTimer, QSize, QLocale,
                QThread, QEvent, QObject, Slot, Signal, Property
            )
            from PySide6.QtWidgets import (
                QApplication, QWidget, QMainWindow, QDialog, QLabel, QMessageBox,
                QVBoxLayout, QHBoxLayout, QPushButton, QToolButton, QFrame, 
                QFileDialog, QInputDialog, QTextEdit, QGridLayout, QLineEdit,
                QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QCheckBox,
                QRadioButton, QMenu, QMenuBar, QSizePolicy, QProgressDialog, 
                QTabWidget, QListWidget, QListWidgetItem, QSplitter, QProgressBar, 
                QScrollArea, QGroupBox, QSpinBox, QButtonGroup, QFormLayout, 
                QTableWidget, QTableWidgetItem
            )
            from PySide6.QtGui import (
                QPixmap, QImage, QColor, QFont, QFontMetrics, QIcon, QBitmap, QPainter,
                QPen, QBrush, QRegion, QIntValidator, QDoubleValidator, QKeySequence, QAction
            )
            
            try:
                import PySide6
                BINDING_VERSION = PySide6.__version__
            except Exception:
                BINDING_VERSION = None
            
            PYQT_AVAILABLE = True
            PYQT_VERSION = 6
            QT_API = "PySide6"
            QT_HAS_POSITION = True
            QT_VERSION = QtCore.qVersion()
            
            ### Compatibility of various PyQt/PySide properties
            class QtCompat:
                AlignCenter = Qt.AlignmentFlag.AlignCenter
                
                AlignLeft = Qt.AlignmentFlag.AlignLeft
                AlignRight = Qt.AlignmentFlag.AlignRight
                AlignHCenter = Qt.AlignmentFlag.AlignHCenter
                AlignJustify = Qt.AlignmentFlag.AlignJustify
                
                AlignTop = Qt.AlignmentFlag.AlignTop
                AlignBottom = Qt.AlignmentFlag.AlignBottom
                AlignVCenter = Qt.AlignmentFlag.AlignVCenter
                AlignBaseline = Qt.AlignmentFlag.AlignBaseline
                
                AbsoluteSize = Qt.SizeMode.AbsoluteSize
                RelativeSize = Qt.SizeMode.RelativeSize
                
                AscendingOrder = Qt.SortOrder.AscendingOrder
                DescendingOrder = Qt.SortOrder.DescendingOrder
                
                IgnoreAspectRatio = Qt.AspectRatioMode.IgnoreAspectRatio
                KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
                KeepAspectRatioByExpanding = Qt.AspectRatioMode.KeepAspectRatioByExpanding
                
                FastTransformation = Qt.TransformationMode.FastTransformation
                SmoothTransformation = Qt.TransformationMode.SmoothTransformation
                
                Horizontal = Qt.Orientation.Horizontal
                Vertical = Qt.Orientation.Vertical
            
            ### BrushStyle compatibility
            class BrushStyle:
                NoBrush = Qt.BrushStyle.NoBrush
                SolidPattern = Qt.BrushStyle.SolidPattern
                Dense1Pattern = Qt.BrushStyle.Dense1Pattern
                Dense2Pattern = Qt.BrushStyle.Dense2Pattern
                Dense3Pattern = Qt.BrushStyle.Dense3Pattern
                Dense4Pattern = Qt.BrushStyle.Dense4Pattern
                Dense5Pattern = Qt.BrushStyle.Dense5Pattern
                Dense6Pattern = Qt.BrushStyle.Dense6Pattern
                Dense7Pattern = Qt.BrushStyle.Dense7Pattern
                
                HorPattern = Qt.BrushStyle.HorPattern
                VerPattern = Qt.BrushStyle.VerPattern
                CrossPattern = Qt.BrushStyle.CrossPattern
                BDiagPattern = Qt.BrushStyle.BDiagPattern
                FDiagPattern = Qt.BrushStyle.FDiagPattern
                DiagCrossPattern = Qt.BrushStyle.DiagCrossPattern
                
                LinearGradientPattern = Qt.BrushStyle.LinearGradientPattern
                ConicalGradientPattern = Qt.BrushStyle.ConicalGradientPattern
                RadialGradientPattern = Qt.BrushStyle.RadialGradientPattern
                TexturePattern = Qt.BrushStyle.TexturePattern
            
            ### CheckState compatibility
            class CheckState:
                Unchecked = Qt.CheckState.Unchecked
                PartiallyChecked = Qt.CheckState.PartiallyChecked
                Checked = Qt.CheckState.Checked
            
            ### CursorShape compatibility
            class CursorShape:
                ArrowCursor = Qt.CursorShape.ArrowCursor
                UpArrowCursor = Qt.CursorShape.UpArrowCursor
                CrossCursor = Qt.CursorShape.CrossCursor
                WaitCursor = Qt.CursorShape.WaitCursor
                
                IBeamCursor = Qt.CursorShape.IBeamCursor
                SizeVerCursor = Qt.CursorShape.SizeVerCursor
                SizeHorCursor = Qt.CursorShape.SizeHorCursor
                SizeBDiagCursor = Qt.CursorShape.SizeBDiagCursor
                SizeFDiagCursor = Qt.CursorShape.SizeFDiagCursor
                SizeAllCursor = Qt.CursorShape.SizeAllCursor
                
                BlankCursor = Qt.CursorShape.BlankCursor
                SplitVCursor = Qt.CursorShape.SplitVCursor
                SplitHCursor = Qt.CursorShape.SplitHCursor
                
                PointingHandCursor = Qt.CursorShape.PointingHandCursor
                ForbiddenCursor = Qt.CursorShape.ForbiddenCursor
                OpenHandCursor = Qt.CursorShape.OpenHandCursor
                ClosedHandCursor = Qt.CursorShape.ClosedHandCursor
                WhatsThisCursor = Qt.CursorShape.WhatsThisCursor
                BusyCursor = Qt.CursorShape.BusyCursor
                
                DragMoveCursor = Qt.CursorShape.DragMoveCursor
                DragCopyCursor = Qt.CursorShape.DragCopyCursor
                DragLinkCursor = Qt.CursorShape.DragLinkCursor
                BitmapCursor = Qt.CursorShape.BitmapCursor
            
            ### QEvent compatibility
            class EventType:
                Clipboard = QEvent.Type.Clipboard
                Close = QEvent.Type.Close
                CursorChange = QEvent.Type.CursorChange
                Enter = QEvent.Type.Enter
                FontChange = QEvent.Type.FontChange
                Leave = QEvent.Type.Leave
                
                Hide = QEvent.Type.Hide
                HideToParent = QEvent.Type.HideToParent
                
                HoverEnter = QEvent.Type.HoverEnter
                HoverLeave = QEvent.Type.HoverLeave
                HoverMove = QEvent.Type.HoverMove
                
                KeyPress = QEvent.Type.KeyPress
                KeyRelease = QEvent.Type.KeyRelease
                
                MouseButtonPress = QEvent.Type.MouseButtonPress
                MouseButtonRelease = QEvent.Type.MouseButtonRelease
                MouseMove = QEvent.Type.MouseMove
                MouseTrackingChange = QEvent.Type.MouseTrackingChange
                
                Paint = QEvent.Type.Paint
                PaletteChange = QEvent.Type.PaletteChange
                
                Resize = QEvent.Type.Resize
                Scroll = QEvent.Type.Scroll
                Shortcut = QEvent.Type.Shortcut
                
                Show = QEvent.Type.Show
                ShowToParent = QEvent.Type.ShowToParent
                StyleChange = QEvent.Type.StyleChange
                
                Timer = QEvent.Type.Timer
                ToolBarChange = QEvent.Type.ToolBarChange
                ToolTip = QEvent.Type.ToolTip
                ToolTipChange = QEvent.Type.ToolTipChange
                
                UpdateLater = QEvent.Type.UpdateLater
                UpdateRequest = QEvent.Type.UpdateRequest
                
                Wheel = QEvent.Type.Wheel
            
            ### Focus Policy compatibility
            class FocusPolicy:
                TabFocus = Qt.FocusPolicy.TabFocus
                ClickFocus = Qt.FocusPolicy.ClickFocus
                StrongFocus = Qt.FocusPolicy.StrongFocus
                WheelFocus = Qt.FocusPolicy.WheelFocus
                NoFocus = Qt.FocusPolicy.NoFocus
            
            ### QFont Compatibility
            class FontWeight:
                ExtraBold = QFont.Weight.ExtraBold
                Black = QFont.Weight.Black
                Bold = QFont.Weight.Bold
                DemiBold = QFont.Weight.DemiBold
                Medium = QFont.Weight.Medium
                Normal = QFont.Weight.Normal
                Light = QFont.Weight.Light
                ExtraLight = QFont.Weight.ExtraLight
                Thin = QFont.Weight.Thin
            
            class FontStyle:
                Normal = QFont.Style.StyleNormal
                Italic = QFont.Style.StyleItalic
                Oblique = QFont.Style.StyleOblique
            
            ### QFrame compatibility
            class FrameShadow:
                Plain = QFrame.Shadow.Plain
                Raised = QFrame.Shadow.Raised
                Sunken = QFrame.Shadow.Sunken
            
            class FrameShape:
                NoFrame = QFrame.Shape.NoFrame
                Box = QFrame.Shape.Box
                Panel = QFrame.Shape.Panel
                StyledPanel = QFrame.Shape.StyledPanel
                HLine = QFrame.Shape.HLine
                VLine = QFrame.Shape.VLine
                WinPanel = QFrame.Shape.WinPanel
            
            ### GlobalColor compatibility
            class GlobalColor:
                White = Qt.GlobalColor.white
                Black = Qt.GlobalColor.black
                Red = Qt.GlobalColor.red
                DarkRed = Qt.GlobalColor.darkRed
                Green = Qt.GlobalColor.green
                DarkGreen = Qt.GlobalColor.darkGreen
                Blue = Qt.GlobalColor.blue
                DarkBlue = Qt.GlobalColor.darkBlue
                Cyan = Qt.GlobalColor.cyan
                DarkCyan = Qt.GlobalColor.darkCyan
                Magenta = Qt.GlobalColor.magenta
                DarkMagenta = Qt.GlobalColor.darkMagenta
                Yellow = Qt.GlobalColor.yellow
                DarkYellow = Qt.GlobalColor.darkYellow
                Gray = Qt.GlobalColor.gray
                DarkGray = Qt.GlobalColor.darkGray
                LightGray = Qt.GlobalColor.lightGray
                Transparent = Qt.GlobalColor.transparent
                Color0 = Qt.GlobalColor.color0
                Color1 = Qt.GlobalColor.color1
            
            ### QImage compatibility
            class ImageFormat:
                Format_RGB32 = QImage.Format.Format_RGB32
                Format_ARGB32 = QImage.Format.Format_ARGB32
                           
                Format_RGB888 = QImage.Format.Format_RGB888
                Format_RGBA8888 = QImage.Format_RGBA8888
                Format_ARGB8565_Premultiplied = QImage.Format.Format_ARGB8565_Premultiplied
                
                Format_RGB666 = QImage.Format.Format_RGB666
                Format_ARGB6666_Premultiplied = QImage.Format.Format_ARGB6666_Premultiplied
                
                Format_RGB555 = QImage.Format.Format_RGB555
                Format_ARGB8555_Premultiplied = QImage.Format.Format_ARGB8555_Premultiplied
                
                Format_RGB16 = QImage.Format.Format_RGB16
                
                Format_RGB444 = QImage.Format.Format_RGB444
                Format_ARGB4444_Premultiplied = QImage.Format.Format_ARGB4444_Premultiplied
                
                Format_Indexed8 = QImage.Format.Format_Indexed8
                
                Format_Invalid = QImage.Format.Format_Invalid
                Format_Mono = QImage.Format.Format_Mono
                Format_MonoLSB = QImage.Format.Format_MonoLSB
            
            ### ItemFlag compatibility
            class ItemFlag:
                NoItemFlags = Qt.ItemFlag.NoItemFlags
                ItemIsSelectable = Qt.ItemFlag.ItemIsSelectable
                ItemIsEditable = Qt.ItemFlag.ItemIsEditable
                ItemIsDragEnabled = Qt.ItemFlag.ItemIsDragEnabled
                ItemIsDropEnabled = Qt.ItemFlag.ItemIsDropEnabled
                ItemIsUserCheckable = Qt.ItemFlag.ItemIsUserCheckable
                ItemIsEnabled = Qt.ItemFlag.ItemIsEnabled
                ItemIsAutoTristate = Qt.ItemFlag.ItemIsAutoTristate
                ItemNeverHasChildren = Qt.ItemFlag.ItemNeverHasChildren
                ItemIsUserTristate = Qt.ItemFlag.ItemIsUserTristate
            
            ### MouseButton compatibility
            class MouseButton:
                NoButton = Qt.MouseButton.NoButton
                AllButtons = Qt.MouseButton.AllButtons
                
                LeftButton = Qt.MouseButton.LeftButton
                RightButton = Qt.MouseButton.RightButton
                MiddleButton = Qt.MouseButton.MiddleButton
                BackButton = Qt.MouseButton.BackButton
                
                ForwardButton = Qt.MouseButton.ForwardButton
                TaskButton = Qt.MouseButton.TaskButton
                XButton1 = Qt.MouseButton.XButton1
                XButton2 = Qt.MouseButton.XButton2
                
                ExtraButton1 = Qt.MouseButton.ExtraButton1
                ExtraButton2 = Qt.MouseButton.ExtraButton2
                ExtraButton3 = Qt.MouseButton.ExtraButton3
                ExtraButton4 = Qt.MouseButton.ExtraButton4
                ExtraButton5 = Qt.MouseButton.ExtraButton5
                ExtraButton6 = Qt.MouseButton.ExtraButton6
                ExtraButton7 = Qt.MouseButton.ExtraButton7
                ExtraButton8 = Qt.MouseButton.ExtraButton8
                ExtraButton9 = Qt.MouseButton.ExtraButton9
                ExtraButton10 = Qt.MouseButton.ExtraButton10
                ExtraButton11 = Qt.MouseButton.ExtraButton11
                ExtraButton12 = Qt.MouseButton.ExtraButton12
                ExtraButton13 = Qt.MouseButton.ExtraButton13
                ExtraButton14 = Qt.MouseButton.ExtraButton14
                ExtraButton15 = Qt.MouseButton.ExtraButton15
                ExtraButton16 = Qt.MouseButton.ExtraButton16
                ExtraButton17 = Qt.MouseButton.ExtraButton17
                ExtraButton18 = Qt.MouseButton.ExtraButton18
                ExtraButton19 = Qt.MouseButton.ExtraButton19
                ExtraButton20 = Qt.MouseButton.ExtraButton20
                ExtraButton21 = Qt.MouseButton.ExtraButton21
                ExtraButton22 = Qt.MouseButton.ExtraButton22
                ExtraButton23 = Qt.MouseButton.ExtraButton23
                ExtraButton24 = Qt.MouseButton.ExtraButton24
            
            ### PenStyle compatibility
            class PenStyle:
                NoPen = Qt.PenStyle.NoPen
                SolidLine = Qt.PenStyle.SolidLine
                DashLine = Qt.PenStyle.DashLine
                DotLine = Qt.PenStyle.DotLine
                DashDotLine = Qt.PenStyle.DashDotLine
                DashDotDotLine = Qt.PenStyle.DashDotDotLine
                CustomDashLine = Qt.PenStyle.CustomDashLine
            
            ### RenderHint compatibility
            class RenderHint:
                Antialiasing = QPainter.RenderHint.Antialiasing
                TextAntialiasing = QPainter.RenderHint.TextAntialiasing
                SmoothPixmapTransform = QPainter.RenderHint.SmoothPixmapTransform
            
            ### ScrollBarPolicy compatibility
            class ScrollBarPolicy:
                ScrollBarAsNeeded = Qt.ScrollBarPolicy.ScrollBarAsNeeded
                ScrollBarAlwaysOff = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
                ScrollBarAlwaysOn = Qt.ScrollBarPolicy.ScrollBarAlwaysOn
            
            ### QSizePolicy compatibility
            class SizePolicy:
                Fixed = QSizePolicy.Policy.Fixed
                Minimum = QSizePolicy.Policy.Minimum
                Maximum = QSizePolicy.Policy.Maximum
                Preferred = QSizePolicy.Policy.Preferred
                Expanding = QSizePolicy.Policy.Expanding
                MinimumExpanding = QSizePolicy.Policy.MinimumExpanding
                Ignored = QSizePolicy.Policy.Ignored
        except ImportError:
            try:
                from PySide2 import QtCore, QtWidgets, QtGui
                from PySide2.QtCore import (
                    Qt, QPoint, QPointF, QRect, QRectF, QTimer, QSize, QLocale,
                    QThread, QEvent, QObject, Slot, Signal, Property
                )
                from PySide2.QtWidgets import (
                    QApplication, QWidget, QMainWindow, QDialog, QLabel, QMessageBox,
                    QVBoxLayout, QHBoxLayout, QPushButton, QToolButton, QFrame, 
                    QFileDialog, QInputDialog, QTextEdit, QGridLayout, QLineEdit,
                    QComboBox, QColorDialog, QSlider, QDoubleSpinBox, QCheckBox,
                    QRadioButton, QMenu, QMenuBar, QSizePolicy, QProgressDialog, 
                    QTabWidget, QListWidget, QListWidgetItem, QSplitter, QProgressBar, 
                    QScrollArea, QAction, QGroupBox, QSpinBox, QButtonGroup, QFormLayout,
                    QTableWidget, QTableWidgetItem
                )
                from PySide2.QtGui import (
                    QPixmap, QImage, QColor, QFont, QFontMetrics, QIcon, QBitmap, QPainter,
                    QPen, QBrush, QRegion, QIntValidator, QDoubleValidator, QKeySequence, QAction
                )
                
                try:
                    import PySide2
                    BINDING_VERSION = PySide2.__version__
                except Exception:
                    BINDING_VERSION = None
                    
                    PYQT_AVAILABLE = True
                    PYQT_VERSION = 5
                    QT_API = "PySide2"
                    QT_HAS_POSITION = False
                    QT_VERSION = QtCore.qVersion()
                
                ### Compatibility of various PyQt/PySide properties
                class QtCompat:
                    AlignCenter = Qt.AlignCenter
                    
                    AlignLeft = Qt.AlignLeft
                    AlignRight = Qt.AlignRight
                    AlignHCenter = Qt.AlignHCenter
                    AlignJustify = Qt.AlignJustify
                    
                    AlignTop = Qt.AlignTop
                    AlignBottom = Qt.AlignBottom
                    AlignVCenter = Qt.AlignVCenter
                    AlignBaseline = Qt.AlignBaseline
                    
                    AbsoluteSize = Qt.AbsoluteSize
                    RelativeSize = Qt.RelativeSize
                    
                    AscendingOrder = Qt.AscendingOrder
                    DescendingOrder = Qt.DescendingOrder
                    
                    IgnoreAspectRatio = Qt.IgnoreAspectRatio
                    KeepAspectRatio = Qt.KeepAspectRatio
                    KeepAspectRatioByExpanding = Qt.KeepAspectRatioByExpanding
                    
                    FastTransformation = Qt.FastTransformation
                    SmoothTransformation = Qt.SmoothTransformation
                    
                    Horizontal = Qt.Horizontal
                    Vertical = Qt.Vertical
                
                ### BrushStyle compatibility
                class BrushStyle:
                    NoBrush = Qt.NoBrush
                    SolidPattern = Qt.SolidPattern
                    Dense1Pattern = Qt.Dense1Pattern
                    Dense2Pattern = Qt.Dense2Pattern
                    Dense3Pattern = Qt.Dense3Pattern
                    Dense4Pattern = Qt.Dense4Pattern
                    Dense5Pattern = Qt.Dense5Pattern
                    Dense6Pattern = Qt.Dense6Pattern
                    Dense7Pattern = Qt.Dense7Pattern
                    
                    HorPattern = Qt.HorPattern
                    VerPattern = Qt.VerPattern
                    CrossPattern = Qt.CrossPattern
                    BDiagPattern = Qt.BDiagPattern
                    FDiagPattern = Qt.FDiagPattern
                    DiagCrossPattern = Qt.DiagCrossPattern
                    
                    LinearGradientPattern = Qt.LinearGradientPattern
                    ConicalGradientPattern = Qt.ConicalGradientPattern
                    RadialGradientPattern = Qt.RadialGradientPattern
                    TexturePattern = Qt.TexturePattern
                
                ### CheckState compatibility
                class CheckState:
                    Unchecked = Qt.Unchecked
                    PartiallyChecked = Qt.PartiallyChecked
                    Checked = Qt.Checked
                
                ### CursorShape compatibility
                class CursorShape:
                    ArrowCursor = Qt.ArrowCursor
                    UpArrowCursor = Qt.UpArrowCursor
                    CrossCursor = Qt.CrossCursor
                    WaitCursor = Qt.WaitCursor
                    
                    IBeamCursor = Qt.IBeamCursor
                    SizeVerCursor = Qt.SizeVerCursor
                    SizeHorCursor = Qt.SizeHorCursor
                    SizeBDiagCursor = Qt.SizeBDiagCursor
                    SizeFDiagCursor = Qt.SizeFDiagCursor
                    SizeAllCursor = Qt.SizeAllCursor
                    
                    BlankCursor = Qt.BlankCursor
                    SplitVCursor = Qt.SplitVCursor
                    SplitHCursor = Qt.SplitHCursor
                    
                    PointingHandCursor = Qt.PointingHandCursor
                    ForbiddenCursor = Qt.ForbiddenCursor
                    OpenHandCursor = Qt.OpenHandCursor
                    ClosedHandCursor = Qt.ClosedHandCursor
                    WhatsThisCursor = Qt.WhatsThisCursor
                    BusyCursor = Qt.BusyCursor
                    
                    DragMoveCursor = Qt.DragMoveCursor
                    DragCopyCursor = Qt.DragCopyCursor
                    DragLinkCursor = Qt.DragLinkCursor
                    BitmapCursor = Qt.BitmapCursor
                
                ### QEvent compatibility
                class EventType:
                    Clipboard = QEvent.Clipboard
                    Close = QEvent.Close
                    CursorChange = QEvent.CursorChange
                    Enter = QEvent.Enter
                    FontChange = QEvent.FontChange
                    Leave = QEvent.Leave
                    
                    Hide = QEvent.Hide
                    HideToParent = QEvent.HideToParent
                    
                    HoverEnter = QEvent.HoverEnter
                    HoverLeave = QEvent.HoverLeave
                    HoverMove = QEvent.HoverMove
                    
                    KeyPress = QEvent.KeyPress
                    KeyRelease = QEvent.KeyRelease
                    
                    MouseButtonPress = QEvent.MouseButtonPress
                    MouseButtonRelease = QEvent.MouseButtonRelease
                    MouseMove = QEvent.MouseMove
                    MouseTrackingChange = QEvent.MouseTrackingChange
                    
                    Paint = QEvent.Paint
                    PaletteChange = QEvent.PaletteChange
                    
                    Resize = QEvent.Resize
                    Scroll = QEvent.Scroll
                    Shortcut = QEvent.Shortcut
                    
                    Show = QEvent.Show
                    ShowToParent = QEvent.ShowToParent
                    StyleChange = QEvent.StyleChange
                    
                    Timer = QEvent.Timer
                    ToolBarChange = QEvent.ToolBarChange
                    ToolTip = QEvent.ToolTip
                    ToolTipChange = QEvent.ToolTipChange
                    
                    UpdateLater = QEvent.UpdateLater
                    UpdateRequest = QEvent.UpdateRequest
                    
                    Wheel = QEvent.Wheel
                
                ### Focus Policy compatibility
                class FocusPolicy:
                    TabFocus = Qt.TabFocus
                    ClickFocus = Qt.ClickFocus
                    StrongFocus = Qt.StrongFocus
                    WheelFocus = Qt.WheelFocus
                    NoFocus = Qt.NoFocus
                
                ### QFont Compatibility
                class FontWeight:
                    ExtraBold = QFont.ExtraBold
                    Black = QFont.Black
                    Bold = QFont.Bold
                    DemiBold = QFont.DemiBold
                    Medium = QFont.Medium
                    Normal = QFont.Normal
                    Light = QFont.Light
                    ExtraLight = QFont.ExtraLight
                    Thin = QFont.Thin
                
                class FontStyle:
                    Normal = QFont.StyleNormal
                    Italic = QFont.StyleItalic
                    Oblique = QFont.StyleOblique
                
                ### QFrame compatibility
                class FrameShadow:
                    Plain = QFrame.Plain
                    Raised = QFrame.Raised
                    Sunken = QFrame.Sunken
                
                class FrameShape:
                    NoFrame = QFrame.NoFrame
                    Box = QFrame.Box
                    Panel = QFrame.Panel
                    StyledPanel = QFrame.StyledPanel
                    HLine = QFrame.HLine
                    VLine = QFrame.VLine
                    WinPanel = QFrame.WinPanel
                
                ### GlobalColor compatibility
                class GlobalColor:
                    White = Qt.white
                    Black = Qt.black
                    Red = Qt.red
                    DarkRed = Qt.darkRed
                    Green = Qt.green
                    DarkGreen = Qt.darkGreen
                    Blue = Qt.blue
                    DarkBlue = Qt.darkBlue
                    Cyan = Qt.cyan
                    DarkCyan = Qt.darkCyan
                    Magenta = Qt.magenta
                    DarkMagenta = Qt.darkMagenta
                    Yellow = Qt.yellow
                    DarkYellow = Qt.darkYellow
                    Gray = Qt.gray
                    DarkGray = Qt.darkGray
                    LightGray = Qt.lightGray
                    Transparent = Qt.transparent
                    Color0 = Qt.color0
                    Color1 = Qt.color1
                
                ### QImage compatibility
                class ImageFormat:
                    Format_RGB32 = QImage.Format_RGB32
                    Format_ARGB32 = QImage.Format_ARGB32
                               
                    Format_RGB888 = QImage.Format_RGB888
                    Format_RGBA8888 = QImage.Format_RGBA8888
                    Format_ARGB8565_Premultiplied = QImage.Format_ARGB8565_Premultiplied
                    
                    Format_RGB666 = QImage.Format_RGB666
                    Format_ARGB6666_Premultiplied = QImage.Format_ARGB6666_Premultiplied
                    
                    Format_RGB555 = QImage.Format_RGB555
                    Format_ARGB8555_Premultiplied = QImage.Format_ARGB8555_Premultiplied
                    
                    Format_RGB16 = QImage.Format_RGB16
                    
                    Format_RGB444 = QImage.Format_RGB444
                    Format_ARGB4444_Premultiplied = QImage.Format_ARGB4444_Premultiplied
                    
                    Format_Indexed8 = QImage.Format_Indexed8
                    
                    Format_Invalid = QImage.Format_Invalid
                    Format_Mono = QImage.Format_Mono
                    Format_MonoLSB = QImage.Format_MonoLSB
                
                ### ItemFlag compatibility
                class ItemFlag:
                    NoItemFlags = Qt.NoItemFlags
                    ItemIsSelectable = Qt.ItemIsSelectable
                    ItemIsEditable = Qt.ItemIsEditable
                    ItemIsDragEnabled = Qt.ItemIsDragEnabled
                    ItemIsDropEnabled = Qt.ItemIsDropEnabled
                    ItemIsUserCheckable = Qt.ItemIsUserCheckable
                    ItemIsEnabled = Qt.ItemIsEnabled
                    ItemIsAutoTristate = Qt.ItemIsAutoTristate
                    ItemNeverHasChildren = Qt.ItemNeverHasChildren
                    ItemIsUserTristate = Qt.ItemIsUserTristate
                
                ### MouseButton compatibility
                class MouseButton:
                    NoButton = Qt.NoButton
                    AllButtons = Qt.AllButtons
                    
                    LeftButton = Qt.LeftButton
                    RightButton = Qt.RightButton
                    MiddleButton = Qt.MiddleButton
                    BackButton = Qt.BackButton
                    
                    ForwardButton = Qt.ForwardButton
                    TaskButton = Qt.TaskButton
                    XButton1 = Qt.XButton1
                    XButton2 = Qt.XButton2
                    
                    ExtraButton1 = Qt.ExtraButton1
                    ExtraButton2 = Qt.ExtraButton2
                    ExtraButton3 = Qt.ExtraButton3
                    ExtraButton4 = Qt.ExtraButton4
                    ExtraButton5 = Qt.ExtraButton5
                    ExtraButton6 = Qt.ExtraButton6
                    ExtraButton7 = Qt.ExtraButton7
                    ExtraButton8 = Qt.ExtraButton8
                    ExtraButton9 = Qt.ExtraButton9
                    ExtraButton10 = Qt.ExtraButton10
                    ExtraButton11 = Qt.ExtraButton11
                    ExtraButton12 = Qt.ExtraButton12
                    ExtraButton13 = Qt.ExtraButton13
                    ExtraButton14 = Qt.ExtraButton14
                    ExtraButton15 = Qt.ExtraButton15
                    ExtraButton16 = Qt.ExtraButton16
                    ExtraButton17 = Qt.ExtraButton17
                    ExtraButton18 = Qt.ExtraButton18
                    ExtraButton19 = Qt.ExtraButton19
                    ExtraButton20 = Qt.ExtraButton20
                    ExtraButton21 = Qt.ExtraButton21
                    ExtraButton22 = Qt.ExtraButton22
                    ExtraButton23 = Qt.ExtraButton23
                    ExtraButton24 = Qt.ExtraButton24
                
                ### PenStyle compatibility
                class PenStyle:
                    NoPen = Qt.PenStyle.NoPen
                    SolidLine = Qt.PenStyle.SolidLine
                    DashLine = Qt.PenStyle.DashLine
                    DotLine = Qt.PenStyle.DotLine
                    DashDotLine = Qt.PenStyle.DashDotLine
                    DashDotDotLine = Qt.PenStyle.DashDotDotLine
                    CustomDashLine = Qt.PenStyle.CustomDashLine
                
                ### RenderHint compatibility
                class RenderHint:
                    Antialiasing = QPainter.Antialiasing
                    TextAntialiasing = QPainter.TextAntialiasing
                    SmoothPixmapTransform = QPainter.SmoothPixmapTransform
                
                ### ScrollBarPolicy compatibility
                class ScrollBarPolicy:
                    ScrollBarAsNeeded = Qt.ScrollBarAsNeeded
                    ScrollBarAlwaysOff = Qt.ScrollBarAlwaysOff
                    ScrollBarAlwaysOn = Qt.ScrollBarAlwaysOn
                
                ### QSizePolicy compatibility
                class SizePolicy:
                    Fixed = QSizePolicy.Fixed
                    Minimum = QSizePolicy.Minimum
                    Maximum = QSizePolicy.Maximum
                    Preferred = QSizePolicy.Preferred
                    Expanding = QSizePolicy.Expanding
                    MinimumExpanding = QSizePolicy.MinimumExpanding
                    Ignored = QSizePolicy.Ignored
            except ImportError:
                print("No compatible version of PyQt/PySide is installed. This program depends on one of the following versions of Qt:")
                print("\t- PyQt5")
                print("\t- PyQt6")
                print("\t- PySide2")
                print("\t- PySide6")


print(f"Qt API: {QT_API} (PyQt{PYQT_VERSION})")

# -----------------------------------------------------------------------------
# Infos globales
# -----------------------------------------------------------------------------

IS_QT6 = QT_API in ("PyQt6", "PySide6")
IS_QT5 = QT_API in ("PyQt5", "PySide2")
IS_PYQT = QT_API in ("PyQt5", "PyQt6")
IS_PYSIDE = QT_API in ("PySide2", "PySide6")


# -----------------------------------------------------------------------------
# Helpers exec / exec_
# -----------------------------------------------------------------------------

def exec_dialog(dialog) -> int:
    """Compat QDialog.exec()/exec_()."""
    return dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()

def exec_menu(menu) -> int:
    """Compat QMenu.exec()/exec_()."""
    return menu.exec() if hasattr(menu, "exec") else menu.exec_()

def exec_app(app) -> int:
    """Compat QApplication.exec()/exec_()."""
    return app.exec() if hasattr(app, "exec") else app.exec_()


__all__ = [
    "NUMPY_VERSION", "NUMPY_AVAILABLE", 
    "PYQT_AVAILABLE", "PYQT_VERSION", "QT_API", "QT_VERSION", "QT_HAS_POSITION",
    "BINDING_VERSION", "IS_QT6", "IS_QT5", "IS_PYQT", "IS_PYSIDE",
    "QtCore", "QtWidgets", "QtGui", "Qt", "QPoint", "QPointF", "QRect",
    "QRectF", "QTimer", "QSize", "QLocale", "QThread", "QEvent", "QObject", 
    "QApplication", "QWidget", "QMainWindow", "QDialog", "QLabel",
    "QMessageBox", "QVBoxLayout", "QHBoxLayout", "QPushButton", 
    "QToolButton", "QFrame", "QFileDialog", "QInputDialog", "QTextEdit",
    "QGridLayout", "QLineEdit", "QComboBox", "QColorDialog", "QSlider",
    "QDoubleSpinBox", "QCheckBox", "QMenu", "QMenuBar", "QSizePolicy",
    "QProgressDialog", "QTabWidget", "QListWidget", "QListWidgetItem",
    "QSplitter", "QProgressBar", "QScrollArea", "QGroupBox", "QPen", 
    "QBrush", "QPixmap", "QImage", "QColor", "QFont", "QFontMetrics", 
    "QIcon", "QBitmap", "QPainter", "QRegion", "QIntValidator", 
    "QDoubleValidator", "QKeySequence", "QAction", "QRadioButton", 
    "QButtonGroup", "QFormLayout", "QTableWidget", "QTableWidgetItem", 
    "Slot", "Signal", "Property", "BrushStyle", "CheckState", "CursorShape", 
    "EventType", "FocusPolicy", "FontWeight", "FontStyle", "FrameShadow", 
    "FrameShape", "GlobalColor", "ImageFormat", "ItemFlag", "MouseButton", 
    "PenStyle", "RenderHint", "ScrollBarPolicy", "SizePolicy", 
    "exec_dialog", "exec_menu", "exec_app"
    ]


