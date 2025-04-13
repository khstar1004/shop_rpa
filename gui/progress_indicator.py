"""Progress step indicator widget"""

from typing import List
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPoint, pyqtProperty
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFontMetrics
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel

class ProgressStepIndicator(QWidget):
    """Widget for displaying step-by-step progress"""
    
    def __init__(self, steps: List[str], parent=None):
        """
        Initialize
        
        Args:
            steps: List of step names
            parent: Parent widget
        """
        super().__init__(parent)
        self.steps = steps
        self.current_step = 0
        self.completed_steps = 0
        self.step_width = 0
        self.step_height = 40
        self.step_labels = []
        self._animation_progress = 0  # For animated transitions
        self._animation = None
        self._animation_target = 0
        
        self.setMinimumHeight(80)
        self.initUI()
        
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Step indicators
        steps_layout = QHBoxLayout()
        
        for i, step_name in enumerate(self.steps):
            step_label = QLabel(step_name)
            step_label.setAlignment(Qt.AlignCenter)
            step_label.setStyleSheet("""
                QLabel {
                    color: #666666;
                    font-size: 12px;
                    font-weight: normal;
                    padding: 4px;
                }
            """)
            self.step_labels.append(step_label)
            steps_layout.addWidget(step_label)
        
        layout.addLayout(steps_layout)
        layout.addStretch()
    
    def get_animation_progress(self):
        return self._animation_progress
    
    def set_animation_progress(self, value):
        self._animation_progress = value
        self.update()
    
    # Define property for animation
    animation_progress = pyqtProperty(float, get_animation_progress, set_animation_progress)
        
    def paintEvent(self, event):
        """Draw step indicators"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Calculate step spacing
        num_steps = len(self.steps)
        if num_steps <= 1:
            return
            
        self.step_width = width / (num_steps - 1)
        mid_height = height / 2
        
        # Draw connecting lines
        for i in range(num_steps - 1):
            x1 = i * self.step_width + 15
            x2 = (i + 1) * self.step_width - 15
            y = mid_height
            
            # Change line color based on completion
            if i < self.completed_steps:
                # Completed connection
                line_pen = QPen(QColor("#4CAF50"), 2)
                painter.setPen(line_pen)
            elif i == self.completed_steps and i < self._animation_target:
                # Animating connection
                line_pen = QPen(QColor("#2196F3"), 2)
                painter.setPen(line_pen)
                # Calculate progress point based on animation
                progress_x = x1 + (x2 - x1) * self._animation_progress
                # Draw line in two parts - completed and incomplete
                painter.drawLine(int(x1), int(y), int(progress_x), int(y))
                
                # Now draw the incomplete part
                line_pen = QPen(QColor("#E0E0E0"), 2)
                line_pen.setDashPattern([5, 5])  # Dotted line
                painter.setPen(line_pen)
                painter.drawLine(int(progress_x), int(y), int(x2), int(y))
                continue
            else:
                # Incomplete connection
                line_pen = QPen(QColor("#E0E0E0"), 2)
                line_pen.setDashPattern([5, 5])  # Dotted line
                painter.setPen(line_pen)
            
            painter.drawLine(int(x1), int(y), int(x2), int(y))
        
        # Draw step circles
        for i in range(num_steps):
            x = i * self.step_width
            y = mid_height
            
            if i < self.completed_steps:  # Completed step
                painter.setBrush(QBrush(QColor("#4CAF50")))  # Green
                painter.setPen(QPen(QColor("#4CAF50"), 2))
                self.step_labels[i].setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 4px;
                    }
                """)
            elif i == self.current_step:  # Current step
                # Use animation progress for smooth transition if animating
                if self._animation and self._animation.state() == QPropertyAnimation.Running:
                    # Blend between white and blue based on animation progress
                    r = int(255 - (255 - 33) * self._animation_progress)
                    g = int(255 - (255 - 150) * self._animation_progress)
                    b = int(255 - (255 - 243) * self._animation_progress)
                    color = QColor(r, g, b)
                    painter.setBrush(QBrush(color))
                    painter.setPen(QPen(QColor("#2196F3"), 2))
                else:
                    painter.setBrush(QBrush(QColor("#2196F3")))  # Blue
                    painter.setPen(QPen(QColor("#2196F3"), 2))
                
                self.step_labels[i].setStyleSheet("""
                    QLabel {
                        color: #2196F3;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 4px;
                    }
                """)
            else:  # Incomplete step
                painter.setBrush(QBrush(Qt.white))
                painter.setPen(QPen(QColor("#E0E0E0"), 2))
                self.step_labels[i].setStyleSheet("""
                    QLabel {
                        color: #666666;
                        font-size: 12px;
                        font-weight: normal;
                        padding: 4px;
                    }
                """)
            
            # Draw circle
            circle_radius = 12  # Slightly larger circles
            painter.drawEllipse(int(x - circle_radius), int(y - circle_radius), 
                               circle_radius * 2, circle_radius * 2)
            
            # Draw step number inside the circle
            painter.setPen(QPen(Qt.white if i <= self.current_step else QColor("#666666"), 2))
            fm = QFontMetrics(self.font())
            text = str(i + 1)
            text_width = fm.horizontalAdvance(text)
            text_height = fm.height()
            painter.drawText(int(x - text_width / 2), int(y + text_height / 3), text)
            
            # Draw check mark for completed steps
            if i < self.completed_steps:
                painter.setPen(QPen(Qt.white, 2))
                painter.drawLine(int(x - 5), int(y), int(x - 2), int(y + 5))
                painter.drawLine(int(x - 2), int(y + 5), int(x + 5), int(y - 5))
    
    def set_current_step(self, step: int):
        """Set current step with animation"""
        if 0 <= step < len(self.steps):
            if step == self.current_step:
                return
                
            # Create animation for smooth transition
            self._animation_target = step
            self._animation = QPropertyAnimation(self, b"animation_progress")
            self._animation.setDuration(500)  # 500ms duration
            self._animation.setStartValue(0.0)
            self._animation.setEndValue(1.0)
            self._animation.setEasingCurve(QEasingCurve.OutCubic)
            self._animation.start()
            
            self.current_step = step
            self.update()
    
    def set_completed_steps(self, steps: int):
        """Set completed steps with animation"""
        if 0 <= steps <= len(self.steps):
            if steps == self.completed_steps:
                return
                
            # Animate the progress
            self._animation_target = steps
            self._animation = QPropertyAnimation(self, b"animation_progress")
            self._animation.setDuration(500)  # 500ms duration
            self._animation.setStartValue(0.0)
            self._animation.setEndValue(1.0)
            self._animation.setEasingCurve(QEasingCurve.OutCubic)
            self._animation.start()
            
            self.completed_steps = steps
            self.update()

    def get_step_description(self, step_index: int) -> str:
        """Get description for a specific step"""
        if 0 <= step_index < len(self.steps):
            return self.steps[step_index]
        return "" 