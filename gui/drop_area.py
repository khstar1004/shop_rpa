"""Drop area widget for file drag and drop functionality"""

import os
from PyQt5.QtCore import Qt, pyqtSignal, QVariantAnimation, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout

from .i18n import translator as tr

class DropArea(QFrame):
    """Drop area for drag-and-drop file selection"""

    clicked = pyqtSignal()  # Signal emitted when the area is clicked
    file_dropped = pyqtSignal(str)  # Signal emitted when a file is dropped

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)  # Enable drop events
        self.setFixedHeight(150)  # Set fixed height for better layout
        self.is_active = False
        self.is_dark_mode = False
        
        # Animation state
        self.pulse_animation = None
        self.current_scale = 1.0
        self.pulse_direction = 1  # 1: expand, -1: shrink
        
        # Layout for icon and text
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # File icon
        self.icon_label = QLabel()
        self.file_icon = QSvgWidget(os.path.join(os.path.dirname(__file__), "assets", "file.svg"))
        self.file_icon.setFixedSize(48, 48)
        
        icon_layout = QHBoxLayout()
        icon_layout.addStretch()
        icon_layout.addWidget(self.file_icon)
        icon_layout.addStretch()
        layout.addLayout(icon_layout)
        
        # Label with hint text
        self.hint_label = QLabel(tr.get_text("file_drag_hint"))
        self.hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hint_label)
        
        # Additional hint label
        self.sub_hint_label = QLabel(tr.get_text("파일을 선택하거나 여기에 드래그하세요"))
        self.sub_hint_label.setAlignment(Qt.AlignCenter)
        self.sub_hint_label.setStyleSheet("color: #999999; font-size: 11px;")
        layout.addWidget(self.sub_hint_label)
        
        # Apply initial style
        self.setStyleSheet(self._get_inactive_style())
        
        # Create animation for hover effect
        self.border_animation = QVariantAnimation()
        self.border_animation.setDuration(300)  # 300ms for smooth transition
        self.border_animation.valueChanged.connect(self._update_border_width)
        
        # Icon pulse animation
        self.pulse_animation = QTimer()
        self.pulse_animation.timeout.connect(self._pulse_icon)
        self.pulse_animation.setInterval(50)  # 50ms interval
        
        # Drop indicator animation
        self.drop_indicator = None

    def _pulse_icon(self):
        """Icon pulse animation"""
        if self.is_active:
            # Pulse animation (1.0 ~ 1.2 scale)
            self.current_scale += 0.01 * self.pulse_direction
            if self.current_scale >= 1.2:
                self.current_scale = 1.2
                self.pulse_direction = -1
            elif self.current_scale <= 1.0:
                self.current_scale = 1.0
                self.pulse_direction = 1
                
            # Adjust icon size
            size = int(48 * self.current_scale)
            self.file_icon.setFixedSize(size, size)

    def set_dark_mode(self, is_dark):
        """Update style for dark/light mode"""
        self.is_dark_mode = is_dark
        self.setStyleSheet(self._get_active_style() if self.is_active else self._get_inactive_style())

    def _get_inactive_style(self):
        """Get style for inactive (normal) state"""
        if self.is_dark_mode:
            return """
                QFrame {
                    border: 2px dashed #5C5C5C;
                    border-radius: 10px;
                    background-color: #2C2C2C;
                    transition: all 0.3s;
                }
                QLabel {
                    color: #B0B0B0;
                    font-size: 14px;
                }
            """
        else:
            return """
                QFrame {
                    border: 2px dashed #BBBBBB;
                    border-radius: 10px;
                    background-color: #F8F8F8;
                    transition: all 0.3s;
                }
                QLabel {
                    color: #666666;
                    font-size: 14px;
                }
            """

    def _get_active_style(self):
        """Get style for active (highlighted) state"""
        if self.is_dark_mode:
            return """
                QFrame {
                    border: 3px dashed #5C9DFF;
                    border-radius: 10px;
                    background-color: #353535;
                    box-shadow: 0 0 10px rgba(92, 157, 255, 0.5);
                    transition: all 0.3s;
                }
                QLabel {
                    color: #FFFFFF;
                    font-size: 14px;
                    font-weight: bold;
                }
            """
        else:
            return """
                QFrame {
                    border: 3px dashed #4285F4;
                    border-radius: 10px;
                    background-color: #F0F8FF;
                    box-shadow: 0 0 10px rgba(66, 133, 244, 0.5);
                    transition: all 0.3s;
                }
                QLabel {
                    color: #4285F4;
                    font-size: 14px;
                    font-weight: bold;
                }
            """

    def _update_border_width(self, value):
        """Update border width during animation"""
        if self.is_dark_mode:
            self.setStyleSheet(f"""
                QFrame {{
                    border: {value}px dashed #5C9DFF;
                    border-radius: 10px;
                    background-color: #353535;
                }}
                QLabel {{
                    color: #FFFFFF;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    border: {value}px dashed #4285F4;
                    border-radius: 10px;
                    background-color: #F0F8FF;
                }}
                QLabel {{
                    color: #4285F4;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)

    def highlight_active(self):
        """Highlight when dragging file over area"""
        self.is_active = True
        
        # Start animation from current state to active state
        self.border_animation.setStartValue(2.0)
        self.border_animation.setEndValue(3.0)
        self.border_animation.start()
        
        # Update hint text
        self.hint_label.setText(tr.get_text("release_to_drop"))
        self.hint_label.setStyleSheet("font-weight: bold;")
        self.sub_hint_label.setText(tr.get_text("여기에서 손을 떼면 파일이 업로드됩니다"))
        
        # Start icon pulse animation
        if not self.pulse_animation.isActive():
            self.pulse_animation.start()

    def highlight_inactive(self):
        """Return to normal state"""
        self.is_active = False
        
        # Animate back to inactive state
        self.border_animation.setStartValue(3.0)
        self.border_animation.setEndValue(2.0)
        self.border_animation.start()
        
        # Restore hint text
        self.hint_label.setText(tr.get_text("file_drag_hint"))
        self.hint_label.setStyleSheet("")
        self.sub_hint_label.setText(tr.get_text("파일을 선택하거나 여기에 드래그하세요"))
        
        # Stop icon pulse animation
        if self.pulse_animation.isActive():
            self.pulse_animation.stop()
            self.file_icon.setFixedSize(48, 48)  # Restore original size

    def mousePressEvent(self, event):
        """Handle mouse press event"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()

    def enterEvent(self, event):
        """Handle mouse enter event for hover effect"""
        if not self.is_active:
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #4285F4;
                    border-radius: 10px;
                    background-color: rgba(66, 133, 244, 0.05);
                }
                QLabel {
                    color: #4285F4;
                    font-size: 14px;
                }
            """)
            
            # Highlight additional hint
            self.sub_hint_label.setStyleSheet("color: #4285F4; font-size: 11px;")

    def leaveEvent(self, event):
        """Handle mouse leave event for hover effect"""
        if not self.is_active:
            self.setStyleSheet(self._get_inactive_style())
            # Restore additional hint
            self.sub_hint_label.setStyleSheet("color: #999999; font-size: 11px;")
            
    def dragEnterEvent(self, event):
        """Handle file drag enter event"""
        # Only allow file drag
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.highlight_active()
            
    def dragLeaveEvent(self, event):
        """Handle file drag leave event"""
        self.highlight_inactive()
        
    def dropEvent(self, event):
        """Handle file drop event"""
        # Get first item from dropped URLs
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isfile(file_path):
                # Emit signal if it's a file
                self.file_dropped.emit(file_path)
                
                # Show drop effect animation
                self._show_drop_effect()
                
        self.highlight_inactive()
        
    def _show_drop_effect(self):
        """Show file drop effect"""
        # Simple check animation using temporary label
        if self.drop_indicator is None:
            self.drop_indicator = QLabel(self)
            self.drop_indicator.setAlignment(Qt.AlignCenter)
            self.drop_indicator.setStyleSheet("""
                background-color: rgba(16, 185, 129, 0.8);
                color: white;
                border-radius: 20px;
                font-size: 24px;
                font-weight: bold;
            """)
            
        # Add check mark and animation
        self.drop_indicator.setText("✓")
        self.drop_indicator.resize(40, 40)
        self.drop_indicator.move(
            (self.width() - self.drop_indicator.width()) // 2,
            (self.height() - self.drop_indicator.height()) // 2
        )
        self.drop_indicator.show()
        
        # Hide after delay
        QTimer.singleShot(1000, self.drop_indicator.hide) 