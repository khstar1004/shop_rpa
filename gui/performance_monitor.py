"""Performance monitor"""

import psutil
import threading
import time
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QTextEdit
)
from .resource_graph import ResourceUsageGraph

class PerformanceMonitor(QObject):
    """Performance monitoring class"""
    
    # Performance data update signal
    data_updated = pyqtSignal(dict)
    # Performance warning signal
    warning = pyqtSignal(str, str)  # message, type(memory, cpu, disk)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.memory_threshold = 80  # MB
        self.cpu_threshold = 80  # %
        self.disk_threshold = 80  # %
        self.interval = 1  # seconds
        self.is_running = False
        self.monitor_thread = None
        
        # History data
        self.history = {
            "memory": [],
            "cpu": [],
            "disk": []
        }
        self.max_history = 60  # 1 minute at 1 second interval
    
    def set_memory_threshold(self, threshold_mb):
        """Set memory usage threshold"""
        self.memory_threshold = threshold_mb
    
    def set_cpu_threshold(self, threshold_percent):
        """Set CPU usage threshold"""
        self.cpu_threshold = threshold_percent
    
    def set_disk_threshold(self, threshold_percent):
        """Set disk usage threshold"""
        self.disk_threshold = threshold_percent
    
    def set_interval(self, interval_seconds):
        """Set monitoring interval"""
        self.interval = interval_seconds
    
    def start(self):
        """Start monitoring"""
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitor_thread)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
    
    def _monitor_thread(self):
        """Monitor thread function"""
        while self.is_running:
            try:
                # Get system metrics
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent()
                disk = psutil.disk_usage('/')
                
                # Convert to MB
                memory_used = memory.used / (1024 * 1024)
                
                # Update history
                self.history["memory"].append(memory_used)
                self.history["cpu"].append(cpu)
                self.history["disk"].append(disk.percent)
                
                # Trim history
                for key in self.history:
                    if len(self.history[key]) > self.max_history:
                        self.history[key] = self.history[key][-self.max_history:]
                
                # Check thresholds
                self._check_warnings(memory_used, cpu, disk.percent)
                
                # Emit data
                data = {
                    "memory": memory_used,
                    "cpu": cpu,
                    "disk": disk.percent,
                    "history": self.history
                }
                self.data_updated.emit(data)
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
            
            time.sleep(self.interval)
    
    def _check_warnings(self, memory_used, cpu, disk):
        """Check for performance warnings"""
        if memory_used > self.memory_threshold:
            self.warning.emit(f"High memory usage: {memory_used:.1f} MB", "memory")
        
        if cpu > self.cpu_threshold:
            self.warning.emit(f"High CPU usage: {cpu:.1f}%", "cpu")
        
        if disk > self.disk_threshold:
            self.warning.emit(f"High disk usage: {disk:.1f}%", "disk")
    
    def get_performance_data(self):
        """Get current performance data"""
        return {
            "memory": self.history["memory"][-1] if self.history["memory"] else 0,
            "cpu": self.history["cpu"][-1] if self.history["cpu"] else 0,
            "disk": self.history["disk"][-1] if self.history["disk"] else 0
        }
    
    def get_history_data(self):
        """Get performance history data"""
        return self.history

class PerformanceMonitorWidget(QWidget):
    """Performance monitor widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.monitor = PerformanceMonitor()
        self.monitor.data_updated.connect(self.update_display)
        self.monitor.warning.connect(self.show_warning)
        self.initUI()
    
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Settings group
        settings_group = QGroupBox("Monitor Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Memory threshold
        self.memory_threshold = QSpinBox()
        self.memory_threshold.setRange(100, 10000)
        self.memory_threshold.setValue(1000)
        self.memory_threshold.setSuffix(" MB")
        self.memory_threshold.valueChanged.connect(
            lambda v: self.monitor.set_memory_threshold(v))
        
        # CPU threshold
        self.cpu_threshold = QSpinBox()
        self.cpu_threshold.setRange(10, 100)
        self.cpu_threshold.setValue(80)
        self.cpu_threshold.setSuffix(" %")
        self.cpu_threshold.valueChanged.connect(
            lambda v: self.monitor.set_cpu_threshold(v))
        
        # Disk threshold
        self.disk_threshold = QSpinBox()
        self.disk_threshold.setRange(10, 100)
        self.disk_threshold.setValue(80)
        self.disk_threshold.setSuffix(" %")
        self.disk_threshold.valueChanged.connect(
            lambda v: self.monitor.set_disk_threshold(v))
        
        # Update interval
        self.update_interval = QSpinBox()
        self.update_interval.setRange(1, 60)
        self.update_interval.setValue(1)
        self.update_interval.setSuffix(" s")
        self.update_interval.valueChanged.connect(
            lambda v: self.monitor.set_interval(v))
        
        # Auto start
        self.auto_start = QCheckBox("Start monitoring on launch")
        self.auto_start.setChecked(True)
        
        settings_layout.addRow("Memory Threshold:", self.memory_threshold)
        settings_layout.addRow("CPU Threshold:", self.cpu_threshold)
        settings_layout.addRow("Disk Threshold:", self.disk_threshold)
        settings_layout.addRow("Update Interval:", self.update_interval)
        settings_layout.addRow("", self.auto_start)
        
        layout.addWidget(settings_group)
        
        # Resource graphs
        self.resource_graph = ResourceUsageGraph()
        layout.addWidget(self.resource_graph)
        
        # Log area
        log_group = QGroupBox("Performance Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(100)
        
        log_layout.addWidget(self.log_area)
        layout.addWidget(log_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.monitor.start)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.monitor.stop)
        self.stop_button.setEnabled(False)
        
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.optimize_performance)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.optimize_button)
        
        layout.addLayout(button_layout)
        
        # Start monitoring if auto-start is enabled
        if self.auto_start.isChecked():
            self.monitor.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
    
    def update_display(self, data):
        """Update display with performance data"""
        # Update resource graph
        self.resource_graph.update_data(data["memory"], data["cpu"])
        
        # Update log
        log_text = (f"Memory: {data['memory']:.1f} MB | "
                   f"CPU: {data['cpu']:.1f}% | "
                   f"Disk: {data['disk']:.1f}%")
        self.log_area.append(log_text)
        
        # Auto-scroll log
        self.log_area.verticalScrollBar().setValue(
            self.log_area.verticalScrollBar().maximum())
    
    def show_warning(self, message, warning_type):
        """Show performance warning"""
        warning_text = f"[{warning_type.upper()}] {message}"
        self.log_area.append(f"<span style='color: red;'>{warning_text}</span>")
    
    def optimize_performance(self):
        """Optimize system performance"""
        # Clear log
        self.log_area.clear()
        self.log_area.append("Optimizing system performance...")
        
        # Add optimization steps here
        # For example:
        # - Clear temporary files
        # - Defragment disk
        # - Adjust process priorities
        # etc.
        
        self.log_area.append("Optimization complete.")
    
    def apply_log_filter(self):
        """Apply log filter"""
        # Implement log filtering if needed
        pass
    
    def closeEvent(self, event):
        """Handle close event"""
        self.monitor.stop()
        event.accept() 