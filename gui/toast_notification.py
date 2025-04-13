"""Toast notification widget"""

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRectF
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtGui import QColor, QPainter, QPainterPath

class ToastNotification(QFrame):
    """Toast notification widget for displaying temporary messages"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
            QLabel {
                color: #333333;
                font-size: 12px;
                padding: 5px;
            }
        """)
        self.initUI()
        
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Message label
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        
        # Set fixed width
        self.setFixedWidth(300)
        
    def show_message(self, message, message_type="info", duration=3000):
        """Show toast message"""
        # Set message text
        self.message_label.setText(message)
        
        # Adjust size
        self.adjustSize()
        
        # Set position (center of parent)
        if self.parent():
            parent_rect = self.parent().geometry()
            x = parent_rect.center().x() - self.width() // 2
            y = parent_rect.bottom() - self.height() - 20
            self.move(x, y)
        
        # Set style based on message type
        if message_type == "success":
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(76, 175, 80, 0.9);
                    border-radius: 8px;
                    border: 1px solid #4CAF50;
                }
                QLabel {
                    color: white;
                    font-size: 12px;
                    padding: 5px;
                }
            """)
        elif message_type == "warning":
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(255, 152, 0, 0.9);
                    border-radius: 8px;
                    border: 1px solid #FF9800;
                }
                QLabel {
                    color: white;
                    font-size: 12px;
                    padding: 5px;
                }
            """)
        elif message_type == "error":
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(244, 67, 54, 0.9);
                    border-radius: 8px;
                    border: 1px solid #F44336;
                }
                QLabel {
                    color: white;
                    font-size: 12px;
                    padding: 5px;
                }
            """)
        else:  # info
            self.setStyleSheet("""
                QFrame {
                    background-color: rgba(255, 255, 255, 0.9);
                    border-radius: 8px;
                    border: 1px solid #E0E0E0;
                }
                QLabel {
                    color: #333333;
                    font-size: 12px;
                    padding: 5px;
                }
            """)
        
        # Show with animation
        self.show()
        self.raise_()
        
        # Fade in animation
        self.setWindowOpacity(0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(300)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()
        
        # Auto hide after duration
        QTimer.singleShot(duration, self.hide_message)
    
    def hide_message(self):
        """Hide toast message with animation"""
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(300)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.setEasingCurve(QEasingCurve.InCubic)
        self.animation.finished.connect(self.hide)
        self.animation.start()
    
    def paintEvent(self, event):
        """Custom paint event for rounded corners"""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Create rounded rectangle path
            path = QPainterPath()
            rect = QRectF(self.rect())  # Convert QRect to QRectF
            path.addRoundedRect(rect, 8, 8)
            
            # Fill with background color
            painter.fillPath(path, QColor(255, 255, 255, 230))
            
            # Draw border
            painter.setPen(QColor(224, 224, 224))
            painter.drawPath(path)
        finally:
            painter.end()  # Always end the painter 