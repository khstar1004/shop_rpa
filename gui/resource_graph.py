"""Resource usage graph widget"""

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtWidgets import QWidget

class ResourceUsageGraph(QWidget):
    """Widget for displaying resource usage graphs (CPU, memory, etc.)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memory_values = [0] * 60  # 1 minute of data
        self.cpu_values = [0] * 60
        self.max_memory = 100  # Maximum value (in MB)
        self.max_cpu = 100  # Percentage
        
        self.setMinimumHeight(100)
        self.setMinimumWidth(200)
        
        # Data update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)  # Update every 1 second
    
    def update_data(self, memory_value: float, cpu_value: float):
        """Update data"""
        # Shift data
        self.memory_values.pop(0)
        self.memory_values.append(memory_value)
        
        self.cpu_values.pop(0)
        self.cpu_values.append(cpu_value)
        
        # Adjust maximum value
        if memory_value > self.max_memory:
            self.max_memory = memory_value * 1.2
        
        self.update()
    
    def update_display(self):
        """Update display"""
        self.update()
    
    def paintEvent(self, event):
        """Draw graph"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor("#F5F5F5"))
        
        # Grid
        painter.setPen(QPen(QColor("#E0E0E0"), 1))
        for i in range(1, 4):
            y = height * i / 4
            painter.drawLine(0, int(y), width, int(y))
        
        # Memory graph
        if any(self.memory_values):
            painter.setPen(QPen(QColor("#2196F3"), 2))
            
            # Draw connecting lines
            path_width = width / (len(self.memory_values) - 1)
            for i in range(len(self.memory_values) - 1):
                x1 = i * path_width
                y1 = height - (self.memory_values[i] / self.max_memory) * height
                x2 = (i + 1) * path_width
                y2 = height - (self.memory_values[i + 1] / self.max_memory) * height
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # CPU graph
        if any(self.cpu_values):
            painter.setPen(QPen(QColor("#FF5722"), 2))
            
            # Draw connecting lines
            path_width = width / (len(self.cpu_values) - 1)
            for i in range(len(self.cpu_values) - 1):
                x1 = i * path_width
                y1 = height - (self.cpu_values[i] / self.max_cpu) * height
                x2 = (i + 1) * path_width
                y2 = height - (self.cpu_values[i + 1] / self.max_cpu) * height
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Labels
        painter.setPen(QColor("#333333"))
        painter.drawText(5, 15, f"Memory: {self.memory_values[-1]:.1f} MB")
        painter.setPen(QColor("#333333"))
        painter.drawText(5, 30, f"CPU: {self.cpu_values[-1]:.1f}%") 