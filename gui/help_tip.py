"""Help tip widget"""

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QColor, QPainter, QPainterPath

class HelpTipWidget(QLabel):
    """Help tip widget for displaying tooltips"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 4px;
                border: 1px solid #E0E0E0;
                color: #333333;
                font-size: 12px;
                padding: 8px;
            }
        """)
        self.setWordWrap(True)
        self.setMaximumWidth(300)
        
        # Hide timer
        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.hide)
        
    def show_tip(self, target_widget, text, duration=5000):
        """Show help tip"""
        # Set text
        self.setText(text)
        
        # Adjust size
        self.adjustSize()
        
        # Calculate position
        if target_widget:
            # Get target widget position
            target_pos = target_widget.mapToGlobal(QPoint(0, 0))
            
            # Position below target widget
            x = target_pos.x() + (target_widget.width() - self.width()) // 2
            y = target_pos.y() + target_widget.height() + 5
            
            # Ensure tip is within screen bounds
            screen_rect = self.screen().availableGeometry()
            if x < screen_rect.left():
                x = screen_rect.left()
            elif x + self.width() > screen_rect.right():
                x = screen_rect.right() - self.width()
            
            if y + self.height() > screen_rect.bottom():
                y = target_pos.y() - self.height() - 5
            
            self.move(x, y)
        
        # Show with animation
        self.show()
        self.raise_()
        
        # Fade in animation
        self.setWindowOpacity(0)
        self.animation = QPropertyAnimation(self, b"windowOpacity")
        self.animation.setDuration(200)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        self.animation.start()
        
        # Start hide timer
        self.hide_timer.start(duration)
    
    def paintEvent(self, event):
        """Custom paint event for rounded corners"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(self.rect(), 4, 4)
        
        # Fill with background color
        painter.fillPath(path, QColor(255, 255, 255, 242))
        
        # Draw border
        painter.setPen(QColor(224, 224, 224))
        painter.drawPath(path) 