"""Settings tab widget"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QComboBox, QListWidget, QInputDialog, QMessageBox,
    QStackedWidget
)

class SettingsTabWidget(QWidget):
    """Improved settings tab widget"""
    
    settings_changed = pyqtSignal(str, object)  # Signal emitted when settings change
    profile_changed = pyqtSignal(str)  # Signal emitted when profile changes
    
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.current_category = None
        self.current_widgets = {}  # Currently displayed setting widgets
        self.initUI()
    
    def initUI(self):
        """Initialize UI"""
        main_layout = QHBoxLayout(self)
        
        # Left category list
        category_layout = QVBoxLayout()
        category_layout.setSpacing(0)
        
        # Category header
        category_header = QLabel("Settings Categories")
        category_header.setStyleSheet("""
            font-weight: bold;
            font-size: 14px;
            padding: 10px;
            background-color: #F5F5F5;
            border-bottom: 1px solid #E0E0E0;
        """)
        category_layout.addWidget(category_header)
        
        # Category buttons
        self.category_buttons = {}
        categories = self.settings.get_categories()
        
        for category_name, category in categories.items():
            button = QPushButton(category.display_name)
            button.setProperty("category", category_name)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 10px;
                    border: none;
                    border-bottom: 1px solid #E0E0E0;
                    background-color: white;
                }
                QPushButton:checked {
                    background-color: #E3F2FD;
                    font-weight: bold;
                    color: #1976D2;
                    border-left: 3px solid #1976D2;
                }
                QPushButton:hover:!checked {
                    background-color: #F5F5F5;
                }
            """)
            button.clicked.connect(self.on_category_clicked)
            category_layout.addWidget(button)
            self.category_buttons[category_name] = button
        
        # Profile management button
        profile_button = QPushButton("Profile Management")
        profile_button.setProperty("category", "profiles")
        profile_button.setCheckable(True)
        profile_button.setAutoExclusive(True)
        profile_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 10px;
                border: none;
                border-bottom: 1px solid #E0E0E0;
                background-color: white;
            }
            QPushButton:checked {
                background-color: #E3F2FD;
                font-weight: bold;
                color: #1976D2;
                border-left: 3px solid #1976D2;
            }
            QPushButton:hover:!checked {
                background-color: #F5F5F5;
            }
        """)
        profile_button.clicked.connect(self.on_category_clicked)
        category_layout.addWidget(profile_button)
        self.category_buttons["profiles"] = profile_button
        
        # Add stretch to category area
        category_layout.addStretch()
        
        # Left category panel
        category_panel = QWidget()
        category_panel.setMaximumWidth(200)
        category_panel.setLayout(category_layout)
        
        # Right settings area
        self.settings_stacked_widget = QStackedWidget()
        
        # Create pages for each setting category
        for category_name, category in categories.items():
            page = self.create_category_page(category)
            self.settings_stacked_widget.addWidget(page)
        
        # Add profile management page
        profiles_page = self.create_profiles_page()
        self.settings_stacked_widget.addWidget(profiles_page)
        
        # Add to main layout
        main_layout.addWidget(category_panel)
        main_layout.addWidget(self.settings_stacked_widget, 3)
        
        # Select initial category
        if self.category_buttons:
            next(iter(self.category_buttons.values())).click()
    
    def create_category_page(self, category):
        """Create category settings page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Category title
        title = QLabel(category.display_name)
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Settings group
        settings_group = QGroupBox()
        settings_layout = QFormLayout(settings_group)
        settings_layout.setSpacing(15)
        
        # Display settings for this category
        for key, setting_info in category.settings.items():
            label = QLabel(setting_info["display_name"])
            
            # Create widget based on setting type
            widget = None
            if setting_info["type"] == "bool":
                widget = QCheckBox()
                widget.setChecked(self.settings.get(key, setting_info["default"]))
                widget.stateChanged.connect(lambda state, k=key: self.on_setting_changed(k, bool(state)))
            
            elif setting_info["type"] == "int":
                widget = QSpinBox()
                widget.setValue(self.settings.get(key, setting_info["default"]))
                if setting_info["options"]:
                    widget.setRange(min(setting_info["options"]), max(setting_info["options"]))
                else:
                    widget.setRange(0, 9999)
                widget.valueChanged.connect(lambda value, k=key: self.on_setting_changed(k, value))
            
            elif setting_info["type"] == "float":
                widget = QDoubleSpinBox()
                widget.setValue(self.settings.get(key, setting_info["default"]))
                widget.setDecimals(2)
                widget.setSingleStep(0.05)
                if setting_info["options"]:
                    widget.setRange(min(setting_info["options"]), max(setting_info["options"]))
                else:
                    widget.setRange(0, 1)
                widget.valueChanged.connect(lambda value, k=key: self.on_setting_changed(k, value))
            
            elif setting_info["type"] == "string":
                widget = QLineEdit()
                widget.setText(self.settings.get(key, setting_info["default"]))
                widget.textChanged.connect(lambda value, k=key: self.on_setting_changed(k, value))
            
            elif setting_info["type"] == "choice":
                widget = QComboBox()
                if setting_info["options"]:
                    widget.addItems(setting_info["options"])
                    current_value = self.settings.get(key, setting_info["default"])
                    try:
                        index = setting_info["options"].index(current_value)
                        widget.setCurrentIndex(index)
                    except ValueError:
                        widget.setCurrentIndex(0)
                widget.currentTextChanged.connect(lambda value, k=key: self.on_setting_changed(k, value))
            
            if widget:
                # Set tooltip
                if setting_info["description"]:
                    widget.setToolTip(setting_info["description"])
                    label.setToolTip(setting_info["description"])
                
                settings_layout.addRow(label, widget)
                self.current_widgets[key] = widget
        
        layout.addWidget(settings_group)
        
        # Help area
        help_label = QLabel("Hover over settings to see help text.")
        help_label.setStyleSheet("color: #757575; font-style: italic;")
        layout.addWidget(help_label)
        
        # Button area
        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(lambda: self.reset_category(category.name))
        button_layout.addStretch()
        button_layout.addWidget(reset_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return page
    
    def create_profiles_page(self):
        """Create profile management page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # Title
        title = QLabel("Profile Management")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Current profile selection
        profile_layout = QHBoxLayout()
        profile_label = QLabel("Current Profile:")
        self.profile_combo = QComboBox()
        self.update_profile_combo()
        
        self.profile_combo.currentTextChanged.connect(self.on_profile_selected)
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.profile_combo, 1)
        layout.addLayout(profile_layout)
        
        # Profile list
        list_group = QGroupBox("Saved Profiles")
        list_layout = QVBoxLayout(list_group)
        
        self.profile_list = QListWidget()
        self.update_profile_list()
        list_layout.addWidget(self.profile_list)
        
        layout.addWidget(list_group)
        
        # Profile management buttons
        button_layout = QHBoxLayout()
        
        new_button = QPushButton("Create New Profile")
        new_button.clicked.connect(self.on_new_profile)
        
        save_button = QPushButton("Save Current Settings")
        save_button.clicked.connect(self.on_save_profile)
        
        delete_button = QPushButton("Delete Profile")
        delete_button.clicked.connect(self.on_delete_profile)
        
        button_layout.addWidget(new_button)
        button_layout.addWidget(save_button)
        button_layout.addWidget(delete_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return page
    
    def update_profile_combo(self):
        """Update profile combo box"""
        self.profile_combo.clear()
        profiles = self.settings.get_all_profiles()
        for name in profiles.keys():
            self.profile_combo.addItem(name)
        
        current_profile = self.settings.get("current_profile", "default")
        index = self.profile_combo.findText(current_profile)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
    
    def update_profile_list(self):
        """Update profile list widget"""
        self.profile_list.clear()
        profiles = self.settings.get_all_profiles()
        for name, profile in profiles.items():
            item = QListWidgetItem(f"{name} - {profile.description}")
            item.setData(Qt.UserRole, name)
            self.profile_list.addItem(item)
    
    def on_category_clicked(self):
        """Handle category button click"""
        sender = self.sender()
        category = sender.property("category")
        self.current_category = category
        
        # Find stack widget index for this category
        categories = list(self.settings.get_categories().keys())
        if category == "profiles":
            index = len(categories)  # Profile page is last
        else:
            index = categories.index(category)
        
        self.settings_stacked_widget.setCurrentIndex(index)
    
    def on_setting_changed(self, key, value):
        """Handle setting value change"""
        self.settings.set(key, value)
        self.settings_changed.emit(key, value)
    
    def reset_category(self, category_name):
        """Reset category settings to defaults"""
        category = self.settings.get_category(category_name)
        if not category:
            return
        
        # Reset all settings in category to defaults
        for key, setting_info in category.settings.items():
            default_value = setting_info["default"]
            self.settings.set(key, default_value)
            
            # Update UI
            if key in self.current_widgets:
                widget = self.current_widgets[key]
                
                if isinstance(widget, QCheckBox):
                    widget.setChecked(default_value)
                elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    widget.setValue(default_value)
                elif isinstance(widget, QLineEdit):
                    widget.setText(default_value)
                elif isinstance(widget, QComboBox) and setting_info["options"]:
                    try:
                        index = setting_info["options"].index(default_value)
                        widget.setCurrentIndex(index)
                    except ValueError:
                        widget.setCurrentIndex(0)
    
    def on_profile_selected(self, profile_name):
        """Handle profile selection"""
        if profile_name and self.settings.apply_profile(profile_name):
            self.profile_changed.emit(profile_name)
    
    def on_new_profile(self):
        """Create new profile"""
        name, ok = QInputDialog.getText(self, "New Profile", "Profile Name:", QLineEdit.Normal, "")
        if ok and name:
            description, ok = QInputDialog.getText(self, "New Profile", "Profile Description:", QLineEdit.Normal, "")
            if ok:
                if self.settings.create_profile(name, description):
                    self.update_profile_combo()
                    self.update_profile_list()
                    # Select new profile
                    index = self.profile_combo.findText(name)
                    if index >= 0:
                        self.profile_combo.setCurrentIndex(index)
    
    def on_save_profile(self):
        """Save current settings as profile"""
        name, ok = QInputDialog.getText(self, "Save Profile", "Profile Name:", QLineEdit.Normal, "")
        if ok and name:
            description, ok = QInputDialog.getText(self, "Save Profile", "Profile Description:", QLineEdit.Normal, "")
            if ok:
                if name in self.settings.get_all_profiles():
                    # Update existing profile
                    self.settings.update_profile(name, description, self.settings.get_all_settings())
                else:
                    # Create new profile
                    self.settings.create_profile(name, description, self.settings.get_all_settings())
                
                self.update_profile_combo()
                self.update_profile_list()
                # Select saved profile
                index = self.profile_combo.findText(name)
                if index >= 0:
                    self.profile_combo.setCurrentIndex(index)
    
    def on_delete_profile(self):
        """Delete profile"""
        selected = self.profile_list.currentItem()
        if selected:
            profile_name = selected.data(Qt.UserRole)
            if profile_name == "default":
                QMessageBox.warning(self, "Warning", "Cannot delete default profile.")
                return
            
            reply = QMessageBox.question(self, "Delete Profile", 
                                        f"Are you sure you want to delete '{profile_name}' profile?",
                                        QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                if self.settings.delete_profile(profile_name):
                    self.update_profile_combo()
                    self.update_profile_list() 