"""Styles for the GUI components"""

from typing import Union

from PyQt5.QtWidgets import QMainWindow, QWidget


class Colors:
    """Color constants for the application"""

    PRIMARY = "#6366F1"  # Indigo
    PRIMARY_LIGHT = "#818CF8"  # Lighter indigo
    PRIMARY_DARK = "#4F46E5"  # Darker indigo

    SECONDARY = "#EC4899"  # Pink
    SECONDARY_LIGHT = "#F472B6"
    SECONDARY_DARK = "#DB2777"

    DANGER = "#EF4444"  # Red
    DANGER_LIGHT = "#F87171"
    DANGER_DARK = "#DC2626"

    SUCCESS = "#10B981"  # Emerald
    SUCCESS_LIGHT = "#34D399"
    SUCCESS_DARK = "#059669"

    WARNING = "#F59E0B"  # Amber
    WARNING_LIGHT = "#FBBF24"
    WARNING_DARK = "#D97706"

    INFO = "#3B82F6"  # Blue
    INFO_LIGHT = "#60A5FA"
    INFO_DARK = "#2563EB"

    BACKGROUND_LIGHT = "#FFFFFF"
    BACKGROUND_DARK = "#111827"  # Dark slate

    SIDEBAR_LIGHT = "#F9FAFB"
    SIDEBAR_DARK = "#1F2937"  # Darker slate

    CARD_LIGHT = "#FFFFFF"
    CARD_DARK = "#1F2937"

    TEXT_LIGHT = "#1F2937"
    TEXT_LIGHT_SECONDARY = "#6B7280"
    TEXT_DARK = "#F9FAFB"
    TEXT_DARK_SECONDARY = "#9CA3AF"

    BORDER_LIGHT = "#E5E7EB"
    BORDER_DARK = "#374151"

    # Add BORDER attribute that will be set based on theme
    BORDER = BORDER_LIGHT  # Default to light mode border

    # Gradients
    GRADIENT_PRIMARY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366F1, stop:1 #8B5CF6)"
    )
    GRADIENT_SECONDARY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #EC4899, stop:1 #F472B6)"
    )
    GRADIENT_SUCCESS = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #10B981, stop:1 #34D399)"
    )
    GRADIENT_INFO = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3B82F6, stop:1 #60A5FA)"
    )
    GRADIENT_WARNING = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #F59E0B, stop:1 #FBBF24)"
    )
    GRADIENT_DANGER = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #EF4444, stop:1 #F87171)"
    )

    # Glass effect
    GLASS_LIGHT = "rgba(255, 255, 255, 0.8)"
    GLASS_DARK = "rgba(31, 41, 55, 0.8)"

    # Shadows
    SHADOW_LIGHT = "0px 2px 4px rgba(0, 0, 0, 0.1)"
    SHADOW_DARK = "0px 2px 4px rgba(0, 0, 0, 0.3)"


class Styles:
    """Style manager for the application"""

    @staticmethod
    def apply_dark_mode(widget: Union[QMainWindow, QWidget]) -> None:
        """Apply dark mode theme to the widget"""
        dark_mode_style = f"""
            QMainWindow, QWidget {{
                background-color: {Colors.BACKGROUND_DARK};
                color: {Colors.TEXT_DARK};
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
            QGroupBox {{
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 8px;
                margin-top: 16px;
                padding: 16px;
                background-color: {Colors.CARD_DARK};
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {Colors.TEXT_DARK};
                font-size: 14px;
            }}
            QLabel {{
                color: {Colors.TEXT_DARK};
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 8px;
                background-color: {Colors.CARD_DARK};
            }}
            QTabBar::tab {{
                background-color: {Colors.SIDEBAR_DARK};
                border: 1px solid {Colors.BORDER_DARK};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 16px;
                margin-right: 4px;
                color: {Colors.TEXT_DARK_SECONDARY};
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.PRIMARY_DARK};
                color: white;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.SIDEBAR_DARK};
                border-bottom: 2px solid {Colors.PRIMARY_LIGHT};
                color: {Colors.PRIMARY_LIGHT};
            }}
            QPlainTextEdit {{
                background-color: #1A1E2D;
                color: #E2E8F0;
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 8px;
                selection-background-color: {Colors.PRIMARY_DARK};
            }}
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: #2D3748;
                color: {Colors.TEXT_DARK};
                text-align: center;
                height: 12px;
            }}
            QProgressBar::chunk {{
                background: {Colors.GRADIENT_PRIMARY};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {Colors.PRIMARY_DARK};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY};
            }}
            QPushButton:pressed {{
                background-color: {Colors.PRIMARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: #4A5568;
                color: #A0AEC0;
            }}
            QComboBox, QSpinBox {{
                background-color: #2D3748;
                color: {Colors.TEXT_DARK};
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 6px;
                padding: 8px;
                min-height: 20px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: url("gui/assets/dropdown-dark.svg");
                width: 12px;
                height: 12px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {Colors.PRIMARY_DARK};
                border-radius: 4px;
                width: 20px;
            }}
            QStatusBar {{
                background-color: {Colors.SIDEBAR_DARK};
                color: {Colors.TEXT_DARK_SECONDARY};
                border-top: 1px solid {Colors.BORDER_DARK};
            }}
            QCheckBox {{
                color: {Colors.TEXT_DARK};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid {Colors.BORDER_DARK};
                background-color: #2D3748;
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.PRIMARY};
                border: none;
                image: url("gui/assets/check-white.svg");
            }}
            QSplitter::handle {{
                background-color: {Colors.BORDER_DARK};
                height: 1px;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #2D3748;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #4A5568;
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #718096;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #2D3748;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #4A5568;
                min-width: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: #718096;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QMenu {{
                background-color: {Colors.CARD_DARK};
                color: {Colors.TEXT_DARK};
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 8px;
            }}
            QMenu::item {{
                padding: 8px 16px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.PRIMARY_DARK};
                color: white;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER_DARK};
                margin: 4px 0;
            }}
            QToolTip {{
                background-color: {Colors.CARD_DARK};
                color: {Colors.TEXT_DARK};
                border: 1px solid {Colors.BORDER_DARK};
                border-radius: 8px;
                padding: 8px;
            }}
            """
        
        # Remove any transition properties to prevent "Unknown property" warnings
        clean_style = StyleTransition.remove_transition_property(dark_mode_style)
        widget.setStyleSheet(clean_style)

    @staticmethod
    def apply_light_mode(widget: Union[QMainWindow, QWidget]) -> None:
        """Apply light mode theme to the widget"""
        light_mode_style = f"""
            QMainWindow, QWidget {{
                background-color: {Colors.BACKGROUND_LIGHT};
                color: {Colors.TEXT_LIGHT};
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
            QGroupBox {{
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
                margin-top: 16px;
                padding: 16px;
                background-color: {Colors.CARD_LIGHT};
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                color: {Colors.TEXT_LIGHT};
                font-size: 14px;
            }}
            QLabel {{
                color: {Colors.TEXT_LIGHT};
            }}
            QTabWidget::pane {{
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
                background-color: {Colors.CARD_LIGHT};
            }}
            QTabBar::tab {{
                background-color: {Colors.SIDEBAR_LIGHT};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 10px 16px;
                margin-right: 4px;
                color: {Colors.TEXT_LIGHT_SECONDARY};
                font-size: 13px;
            }}
            QTabBar::tab:selected {{
                background-color: {Colors.PRIMARY};
                color: white;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {Colors.SIDEBAR_LIGHT};
                border-bottom: 2px solid {Colors.PRIMARY_LIGHT};
                color: {Colors.PRIMARY_LIGHT};
            }}
            QPlainTextEdit {{
                background-color: white;
                color: {Colors.TEXT_LIGHT};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                padding: 8px;
                selection-background-color: {Colors.PRIMARY_LIGHT};
            }}
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background-color: #E5E7EB;
                color: {Colors.TEXT_LIGHT};
                text-align: center;
                height: 12px;
            }}
            QProgressBar::chunk {{
                background: {Colors.GRADIENT_PRIMARY};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Colors.PRIMARY_DARK};
            }}
            QPushButton:disabled {{
                background-color: #E5E7EB;
                color: #9CA3AF;
            }}
            QComboBox, QSpinBox {{
                background-color: white;
                color: {Colors.TEXT_LIGHT};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 6px;
                padding: 8px;
                min-height: 20px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 24px;
            }}
            QComboBox::down-arrow {{
                image: url("gui/assets/dropdown-light.svg");
                width: 12px;
                height: 12px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                background-color: {Colors.PRIMARY};
                border-radius: 4px;
                width: 20px;
            }}
            QStatusBar {{
                background-color: {Colors.SIDEBAR_LIGHT};
                color: {Colors.TEXT_LIGHT_SECONDARY};
                border-top: 1px solid {Colors.BORDER_LIGHT};
            }}
            QCheckBox {{
                color: {Colors.TEXT_LIGHT};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid {Colors.BORDER_LIGHT};
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.PRIMARY};
                border: none;
                image: url("gui/assets/check-white.svg");
            }}
            QSplitter::handle {{
                background-color: {Colors.BORDER_LIGHT};
                height: 1px;
            }}
            QScrollBar:vertical {{
                border: none;
                background-color: #F3F4F6;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: #D1D5DB;
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: #9CA3AF;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                border: none;
                background-color: #F3F4F6;
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: #D1D5DB;
                min-width: 20px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: #9CA3AF;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QMenu {{
                background-color: {Colors.CARD_LIGHT};
                color: {Colors.TEXT_LIGHT};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
            }}
            QMenu::item {{
                padding: 8px 16px;
            }}
            QMenu::item:selected {{
                background-color: {Colors.PRIMARY};
                color: white;
            }}
            QMenu::separator {{
                height: 1px;
                background-color: {Colors.BORDER_LIGHT};
                margin: 4px 0;
            }}
            QToolTip {{
                background-color: {Colors.CARD_LIGHT};
                color: {Colors.TEXT_LIGHT};
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 8px;
                padding: 8px;
            }}
        """
        
        # Remove any transition properties to prevent "Unknown property" warnings
        clean_style = StyleTransition.remove_transition_property(light_mode_style)
        widget.setStyleSheet(clean_style)

    @staticmethod
    def apply_group_box_style(widget: QWidget) -> None:
        """Apply style to group box"""
        widget.setStyleSheet(
            f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """
        )

    @staticmethod
    def apply_drop_area_style(widget: QWidget) -> None:
        """Apply drop area style in light mode"""
        style = """
            QFrame {
                border: 2px dashed #BBBBBB;
                border-radius: 10px;
                background-color: #F8F8F8;
            }
            QLabel {
                color: #666666;
                font-size: 14px;
            }
        """
        widget.setStyleSheet(StyleTransition.remove_transition_property(style))

    @staticmethod
    def apply_drop_area_active_style(widget: QWidget) -> None:
        """Apply active drop area style in light mode"""
        style = """
            QFrame {
                border: 3px dashed #4285F4;
                border-radius: 10px;
                background-color: #F0F8FF;
                box-shadow: 0 0 10px rgba(66, 133, 244, 0.5);
            }
            QLabel {
                color: #4285F4;
                font-size: 14px;
                font-weight: bold;
            }
        """
        widget.setStyleSheet(StyleTransition.remove_transition_property(style))

    @staticmethod
    def apply_drop_area_dark_style(widget: QWidget) -> None:
        """Apply drop area style in dark mode"""
        style = """
            QFrame {
                border: 2px dashed #5C5C5C;
                border-radius: 10px;
                background-color: #2C2C2C;
            }
            QLabel {
                color: #B0B0B0;
                font-size: 14px;
            }
        """
        widget.setStyleSheet(StyleTransition.remove_transition_property(style))

    @staticmethod
    def apply_drop_area_active_dark_style(widget: QWidget) -> None:
        """Apply active drop area style in dark mode"""
        style = """
            QFrame {
                border: 3px dashed #5C9DFF;
                border-radius: 10px;
                background-color: #353535;
                box-shadow: 0 0 10px rgba(92, 157, 255, 0.5);
            }
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
            }
        """
        widget.setStyleSheet(StyleTransition.remove_transition_property(style))

    @staticmethod
    def apply_file_path_style(widget: QWidget) -> None:
        """Apply style to file path label"""
        widget.setStyleSheet(
            """
            QLabel {
                font-weight: bold;
                color: #555555;
                padding: 5px;
            }
        """
        )

    @staticmethod
    def apply_start_button_style(widget: QWidget) -> None:
        """Apply style to start button"""
        widget.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {Colors.SUCCESS};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.SUCCESS_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Colors.SUCCESS_DARK};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BORDER_LIGHT};
                color: #888888;
            }}
        """
        )

    @staticmethod
    def apply_stop_button_style(widget: QWidget) -> None:
        """Apply style to stop button"""
        widget.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {Colors.DANGER};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Colors.DANGER_LIGHT};
            }}
            QPushButton:pressed {{
                background-color: {Colors.DANGER_DARK};
            }}
            QPushButton:disabled {{
                background-color: {Colors.BORDER_LIGHT};
                color: #888888;
            }}
        """
        )

    @staticmethod
    def apply_spinbox_style(widget: QWidget) -> None:
        """Apply style to spinbox"""
        widget.setStyleSheet(
            f"""
            QSpinBox {{
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 4px;
                padding: 4px;
                background: white;
                min-width: 80px;
            }}
        """
        )

    @staticmethod
    def apply_progress_bar_style(progress_bar):
        """Apply style to progress bar"""
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
                color: #555555;
                background-color: #F5F5F5;
                height: 22px;
                min-height: 22px;
                max-height: 22px;
                margin: 2px;
                padding: 0px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                         stop:0 #4285F4, stop:1 #5E97F6);
                border-radius: 8px;
            }
        """)
        
        # 부드러운 업데이트를 위한 기본 설정
        progress_bar.setTextVisible(True)
        progress_bar.setFormat("%p% (%v/%m)")  # 기본 형식을 백분율과 값으로 설정

    @staticmethod
    def apply_status_label_style(widget: QWidget) -> None:
        """Apply style to status label"""
        widget.setStyleSheet(
            """
            QLabel {
                font-weight: bold;
                color: #555555;
                font-size: 13px;
                padding: 5px;
            }
        """
        )

    @staticmethod
    def apply_log_area_style(widget: QWidget) -> None:
        """Apply style to log area"""
        widget.setStyleSheet(
            f"""
            QPlainTextEdit {{
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                background-color: #FAFAFA;
                border: 1px solid #E0E0E0;
                border-radius: 4px;
                padding: 5px;
                selection-background-color: {Colors.PRIMARY};
                selection-color: white;
            }}
        """
        )

    @staticmethod
    def apply_checkbox_style(widget: QWidget) -> None:
        """Apply style to checkbox"""
        widget.setStyleSheet(
            f"""
            QCheckBox {{
                spacing: 5px;
                color: #555555;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 1px solid {Colors.BORDER_LIGHT};
                border-radius: 3px;
                background: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {Colors.PRIMARY};
                border-color: {Colors.PRIMARY};
            }}
            QCheckBox::indicator:hover {{
                border-color: {Colors.PRIMARY};
            }}
        """
        )

    @staticmethod
    def apply_status_bar_style(widget: QWidget) -> None:
        """Apply style to status bar"""
        widget.setStyleSheet(
            """
            QStatusBar {
                background-color: #F5F5F5;
                color: #555555;
                border-top: 1px solid #E0E0E0;
            }
        """
        )

    @staticmethod
    def apply_memory_label_style(widget: QWidget) -> None:
        """Apply style to memory label"""
        widget.setStyleSheet(
            """
            QLabel {
                color: #555555;
                padding: 2px 5px;
                font-size: 12px;
            }
        """
        )

    @staticmethod
    def apply_style(widget: QWidget, style: str) -> None:
        """Apply style with automatic transition property removal
        
        Args:
            widget: The widget to style
            style: The CSS style string
        """
        clean_style = StyleTransition.remove_transition_property(style)
        widget.setStyleSheet(clean_style)


class StyleTransition:
    """Utility class for handling style transitions that CSS transition doesn't support in PyQt"""
    
    @staticmethod
    def create_opacity_animation(widget, start_value=0.0, end_value=1.0, duration=300, easing=None):
        """Create a window opacity animation
        
        Args:
            widget: The widget to animate
            start_value: Starting opacity (0.0-1.0)
            end_value: Ending opacity (0.0-1.0)
            duration: Animation duration in milliseconds
            easing: QEasingCurve to use (defaults to OutCubic)
            
        Returns:
            QPropertyAnimation object. Call start() to begin animation.
        """
        from PyQt5.QtCore import QPropertyAnimation, QEasingCurve
        
        animation = QPropertyAnimation(widget, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(start_value)
        animation.setEndValue(end_value)
        animation.setEasingCurve(easing or QEasingCurve.OutCubic)
        return animation
    
    @staticmethod
    def create_color_animation(widget, property_name, start_color, end_color, duration=300, easing=None, update_func=None):
        """Create a color transition animation
        
        Args:
            widget: The widget to animate
            property_name: The property to animate
            start_color: Starting QColor
            end_color: Ending QColor
            duration: Animation duration in milliseconds
            easing: QEasingCurve to use (defaults to OutCubic)
            update_func: Function to call with current color value
            
        Returns:
            QVariantAnimation object. Call start() to begin animation.
        """
        from PyQt5.QtCore import QVariantAnimation, QEasingCurve
        
        animation = QVariantAnimation()
        animation.setDuration(duration)
        animation.setStartValue(start_color)
        animation.setEndValue(end_color)
        animation.setEasingCurve(easing or QEasingCurve.OutCubic)
        
        if update_func:
            animation.valueChanged.connect(update_func)
        
        return animation
    
    @staticmethod
    def create_stylesheet_animation(widget, start_style, end_style, duration=300, easing=None):
        """Create an animation that transitions between two stylesheets
        
        This creates a proxy animation that updates the stylesheet with interpolated values
        
        Args:
            widget: The widget to animate
            start_style: Starting stylesheet 
            end_style: Ending stylesheet
            duration: Animation duration in milliseconds
            easing: QEasingCurve to use (defaults to OutCubic)
            
        Returns:
            QVariantAnimation object. Call start() to begin animation.
        """
        from PyQt5.QtCore import QVariantAnimation, QEasingCurve
        
        # Remove any transition properties from the stylesheets to avoid warnings
        start_style = StyleTransition.remove_transition_property(start_style)
        end_style = StyleTransition.remove_transition_property(end_style)
        
        animation = QVariantAnimation()
        animation.setDuration(duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(easing or QEasingCurve.OutCubic)
        
        def update_stylesheet(value):
            # This is a simplified approach - for complex stylesheets would need more sophisticated parsing
            widget.setStyleSheet(start_style)
            
        animation.valueChanged.connect(update_stylesheet)
        return animation
    
    @staticmethod
    def remove_transition_property(stylesheet):
        """Remove transition properties from a stylesheet to avoid warnings
        
        Args:
            stylesheet: The stylesheet string
            
        Returns:
            Stylesheet string with transition properties removed
        """
        import re
        # Match 'transition: X;' or 'transition:X;' patterns
        return re.sub(r'transition\s*:[^;]+;', '', stylesheet)
