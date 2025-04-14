"""Tutorial dialog"""

from PyQt5.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QFrame
)
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPainterPath

class TutorialStep:
    """Tutorial step data class"""
    
    def __init__(self, title, description, target_widget=None, image_path=None, callback=None):
        self.title = title
        self.description = description
        self.target_widget = target_widget
        self.image_path = image_path
        self.callback = callback

class TutorialDialog(QDialog):
    """Interactive tutorial dialog"""
    
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.current_step = 0
        self.highlight_widget = None
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QDialog {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        
        self.initUI()
    
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title_label)
        
        # Description
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)
        
        # Image (if any)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_step)
        self.prev_button.setEnabled(False)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_step)
        
        button_layout.addWidget(self.prev_button)
        button_layout.addStretch()
        button_layout.addWidget(self.next_button)
        
        layout.addLayout(button_layout)
        
        # Update to first step
        self.update_step()
    
    def update_step(self):
        """Update UI for current step"""
        if not self.steps:
            return
        
        step = self.steps[self.current_step]
        
        # Update title and description
        self.title_label.setText(step.title)
        self.description_label.setText(step.description)
        
        # Update image
        if step.image_path:
            pixmap = QPixmap(step.image_path)
            self.image_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_label.show()
        else:
            self.image_label.hide()
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_step > 0)
        self.next_button.setText("Finish" if self.current_step == len(self.steps) - 1 else "Next")
        
        # Highlight target widget
        self.highlight_widget = step.target_widget
        if self.highlight_widget:
            self.highlight_widget.raise_()
        
        # Execute callback if any
        if step.callback:
            step.callback()
    
    def next_step(self):
        """Move to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_step()
        else:
            self.accept()
    
    def prev_step(self):
        """Move to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step()
    
    def paintEvent(self, event):
        """Custom paint event for rounded corners"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 8, 8)
        
        # Fill with background color
        painter.fillPath(path, QColor(255, 255, 255, 242))
        
        # Draw border
        painter.setPen(QColor(224, 224, 224))
        painter.drawPath(path) 