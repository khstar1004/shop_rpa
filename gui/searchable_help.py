"""Searchable help widget"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QTextBrowser, QPushButton
)

class SearchableHelpText(QWidget):
    """Searchable help text widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.help_content = {}
        self.initUI()
    
    def initUI(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # Search area
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search help...")
        self.search_input.textChanged.connect(self.search_help)
        
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.search_input.clear())
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(clear_button)
        
        layout.addLayout(search_layout)
        
        # Content area
        content_layout = QHBoxLayout()
        
        # Table of contents
        self.toc_list = QListWidget()
        self.toc_list.currentItemChanged.connect(self.on_section_selected)
        self.toc_list.setMaximumWidth(200)
        
        # Help content
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        
        content_layout.addWidget(self.toc_list)
        content_layout.addWidget(self.content_browser)
        
        layout.addLayout(content_layout)
    
    def load_help_content(self, content_dict):
        """Load help content"""
        self.help_content = content_dict
        self.update_toc()
    
    def update_toc(self):
        """Update table of contents"""
        self.toc_list.clear()
        
        for section in self.help_content.keys():
            item = QListWidgetItem(section)
            item.setData(Qt.UserRole, section)
            self.toc_list.addItem(item)
        
        # Select first item
        if self.toc_list.count() > 0:
            self.toc_list.setCurrentItem(self.toc_list.item(0))
    
    def on_section_selected(self, current, previous):
        """Handle section selection"""
        if current:
            self.show_section(current.data(Qt.UserRole))
    
    def show_section(self, section):
        """Show help section"""
        if section not in self.help_content:
            return
        
        content = self.help_content[section]
        html = f"<h2>{section}</h2>"
        
        # Add text
        if "text" in content:
            html += f"<p>{content['text']}</p>"
        
        # Add links
        if "links" in content and content["links"]:
            html += "<h3>Related Topics</h3><ul>"
            for link_text, link_target in content["links"].items():
                html += f'<li><a href="#{link_target}">{link_text}</a></li>'
            html += "</ul>"
        
        # Add images
        if "images" in content and content["images"]:
            for image_path in content["images"]:
                html += f'<p><img src="{image_path}" style="max-width: 100%;"></p>'
        
        self.content_browser.setHtml(html)
    
    def search_help(self, text):
        """Search help content"""
        if not text:
            self.update_toc()
            return
        
        text = text.lower()
        self.toc_list.clear()
        
        # Search in content
        for section, content in self.help_content.items():
            section_text = section.lower()
            content_text = content.get("text", "").lower()
            
            if text in section_text or text in content_text:
                item = QListWidgetItem(section)
                item.setData(Qt.UserRole, section)
                self.toc_list.addItem(item)
        
        # Select first result
        if self.toc_list.count() > 0:
            self.toc_list.setCurrentItem(self.toc_list.item(0)) 