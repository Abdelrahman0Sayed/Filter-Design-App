import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import pyqtgraph as pg 
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from scipy.signal import zpk2tf, sosfreqz, sos2tf, tf2sos   # Used to convert zeros, poles, and gain to transfer function ,
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import numpy as np
import json
from PyQt5.QtCore import QTimer
from collections import deque
import time
import pyqtgraph as pg
pg.setConfigOptions(antialias=True)

# Dark theme colors
DARK_PRIMARY = "#1e1e1e"
DARK_SECONDARY = "#2d2d2d"
ACCENT_COLOR = "#007acc"
TEXT_COLOR = "#ffffff"
PLOT_BG = "#2d2d2d"
PLOT_TEXT = "#ffffff"
PLOT_GRID = "#404040"


class FilterDesignApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Digital Filter Designer")
        self.setGeometry(100, 100, 1200, 800)

        
        self.setup_toolbar()

        self.conjugate_pairs = {'zeros': {}, 'poles': {}}
        
        # Setup undo/redo
        self.history = []
        self.history_index = -1

        self.dragging = False
        self.drag_target = None
        self.drag_type = None
        self.dragPoint = None
        self.dragOffset = None

        # Initialize filter states
        self.direct_state = None
        self.cascade_state = None
        
        # Initialize all-pass filters
        self.all_pass_filters = []
        self.all_pass_library = AllPassLibrary()
        
        
        
        # Set application style to fusion for better dark theme support
        QApplication.setStyle("Fusion")
        self.setup_dark_palette()
        
        # Initialize filter data
        self.zeros = []
        self.poles = []
        self.current_mode = None
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self.filter_design_tab = QWidget()
        self.real_time_tab = QWidget()
        
        # Add tabs
        self.tabs.addTab(self.filter_design_tab, "Filter Design")
        self.tabs.addTab(self.real_time_tab, "Real-time Processing")
        
        # Setup layouts for each tab
        self.setup_filter_design_tab()
        self.setup_real_time_tab()
        
        
        
        
        
        # Initialize signal processing variables
        from collections import deque
        self.max_samples = 10000
        self.input_signal = deque(maxlen=self.max_samples)
        self.output_signal = deque(maxlen=self.max_samples)
        self.buffer_index = 0
        self.last_time = time.time()
        self.last_mouse_pos = None

        # Remove phase_corrected_signal
        self.processing_speed = 50
        self.last_mouse_y = None

        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_next_sample)
        self.process_timer.start(20)

    def setup_filter_design_tab(self):
        """Setup the filter design tab with z-plane and frequency response"""
        layout = QHBoxLayout()

        # Left panel styling and setup
        left_panel = QGroupBox("Controls")
        left_panel.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {ACCENT_COLOR};
                border-radius: 5px;
                margin-top: 1em;
                padding: 15px;
            }}
            QGroupBox::title {{
                color: {TEXT_COLOR};
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
                font-weight: bold;
            }}
        """)
        
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        # Style buttons
        button_style = f"""
            QPushButton {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 4px;
                padding: 8px;
                min-width: 80px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #005999;
            }}
            QPushButton:checked {{
                background-color: {ACCENT_COLOR};
                color: white;
            }}
        """
        
        self.add_zero_btn = QPushButton("Add Zero")
        self.add_pole_btn = QPushButton("Add Pole")
        self.add_zero_btn.setCheckable(True)
        self.add_pole_btn.setCheckable(True)
        self.clear_all_btn = QPushButton("Clear All")
        self.conjugate_check = QCheckBox("Add Conjugates")
        self.swap = QPushButton("swap zeros & poles")
        self.code = QPushButton("generate C code")
        self.export = QPushButton("export realization")

        self.direct_form = QRadioButton("Direct Form II")
        self.cascade_form = QRadioButton("Cascade Form")
        radio_style = f"""
            QRadioButton {{
                color: {TEXT_COLOR};
                spacing: 5px;
            }}
            QRadioButton::indicator {{
                width: 15px;
                height: 15px;
            }}
            QRadioButton::indicator:unchecked {{
                border: 1px solid {ACCENT_COLOR};
                border-radius: 7px;
                background: {DARK_SECONDARY};
            }}
            QRadioButton::indicator:checked {{
                border: 1px solid {ACCENT_COLOR};
                border-radius: 7px;
                background: {ACCENT_COLOR};
            }}
        """
        self.direct_form.setStyleSheet(radio_style)
        self.cascade_form.setStyleSheet(radio_style)
        
        self.direct_form.setChecked(True)  # Default to Direct Form II

        self.export.clicked.connect(self.export_filter)
        
        for btn in [self.add_zero_btn, self.add_pole_btn, self.clear_all_btn , self.swap,self.code,self.export]:
            btn.setStyleSheet(button_style)
        
        self.conjugate_check.setStyleSheet(f"""
            QCheckBox {{
                color: {TEXT_COLOR};
                spacing: 5px;
            }}
            QCheckBox::indicator {{
                width: 15px;
                height: 15px;
            }}
            QCheckBox::indicator:unchecked {{
                border: 1px solid {ACCENT_COLOR};
                background: {DARK_SECONDARY};
            }}
            QCheckBox::indicator:checked {{
                border: 1px solid {ACCENT_COLOR};
                background: {ACCENT_COLOR};
            }}
        """)
        # Connect signals and add widgets
        self.add_zero_btn.clicked.connect(lambda: self.set_mode('zero'))
        self.add_pole_btn.clicked.connect(lambda: self.set_mode('pole'))
        self.clear_all_btn.clicked.connect(self.clear_all)
        self.swap.clicked.connect(self.swap_zeros_poles)
        self.code.clicked.connect(self.generate_c_code)


        
        left_layout.addWidget(self.add_zero_btn)
        left_layout.addWidget(self.add_pole_btn)
        left_layout.addWidget(self.clear_all_btn)
        left_layout.addWidget(self.conjugate_check)
        left_layout.addWidget(self.swap)
        left_layout.addWidget(self.code)
        left_layout.addWidget(self.export)


        left_layout.addStretch()

            
        # Points Control Group
        points_group = QGroupBox("Points Control")
        points_layout = QVBoxLayout()
        
        # Add Zero/Pole buttons in horizontal layout
        add_points_layout = QHBoxLayout()
        add_points_layout.addWidget(self.add_zero_btn)
        add_points_layout.addWidget(self.add_pole_btn)
        points_layout.addLayout(add_points_layout)
        
        # Clear buttons in horizontal layout
        clear_layout = QHBoxLayout()
        self.clear_zeros_btn = QPushButton("Clear Zeros")
        self.clear_poles_btn = QPushButton("Clear Poles")
        self.clear_zeros_btn.clicked.connect(self.clear_zeros)
        self.clear_poles_btn.clicked.connect(self.clear_poles)

        self.button_style = f"""
            QPushButton {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                border-radius: 4px;
                padding: 8px;
                min-width: 80px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
                color: white;
            }}
            QPushButton:pressed {{
                background-color: #005999;
            }}
            QPushButton:checked {{
                background-color: {ACCENT_COLOR};
                color: white;
            }}
        """
        
        for btn in [self.clear_zeros_btn, self.clear_poles_btn]:
            btn.setStyleSheet(self.button_style)
        
        clear_layout.addWidget(self.clear_zeros_btn)
        clear_layout.addWidget(self.clear_poles_btn)
        points_layout.addLayout(clear_layout)
        
        # Clear All and Swap buttons
        points_layout.addWidget(self.clear_all_btn)
        points_layout.addWidget(self.swap)
        points_layout.addWidget(self.conjugate_check)
        points_group.setLayout(points_layout)
        
        # Implementation Group
        impl_group = QGroupBox("Implementation")
        impl_layout = QVBoxLayout()
        impl_layout.addWidget(self.direct_form)
        impl_layout.addWidget(self.cascade_form)
        impl_layout.addWidget(self.export)
        impl_layout.addWidget(self.code)
        impl_group.setLayout(impl_layout)
        
        # Add all groups to main layout
        left_layout.addWidget(points_group)
        left_layout.addWidget(impl_group)
        left_panel.setLayout(left_layout)
        
        # Update plot styling
        plt.style.use('dark_background')
        
        
        
        
        # Center panel for z-plane
        center_panel = QGroupBox("Z-Plane")
        center_layout = QVBoxLayout()
        self.z_plane_figure = Figure(figsize=(6, 6))
        self.z_plane_canvas = FigureCanvas(self.z_plane_figure)
        center_layout.addWidget(self.z_plane_canvas)
        # Add motion and release connections
        self.z_plane_canvas.mpl_connect('button_press_event', self.on_press)
        self.z_plane_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.z_plane_canvas.mpl_connect('button_release_event', self.on_release)

        self.z_plane_canvas.mpl_disconnect('button_press_event')

        center_panel.setLayout(center_layout)
        
        # Right panel for frequency response
        right_panel = QGroupBox("Frequency Response")
        right_layout = QVBoxLayout()
        self.freq_figure = Figure(figsize=(6, 8))
        self.freq_canvas = FigureCanvas(self.freq_figure)
        
        # Create two subplots for magnitude and phase
        gs = self.freq_figure.add_gridspec(2, 1, height_ratios=[1, 1])
        self.mag_ax = self.freq_figure.add_subplot(gs[0])
        self.phase_ax = self.freq_figure.add_subplot(gs[1])
        
        right_layout.addWidget(self.freq_canvas)
        right_panel.setLayout(right_layout)

        ###################
        all_pass_panel_side = self.setup_all_pass_panel()
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(center_panel)
        layout.addWidget(right_panel)
        layout.addWidget(all_pass_panel_side)
        
        self.initialize_plots()


        
        self.filter_design_tab.setLayout(layout)

    def setup_real_time_tab(self):
        """Setup the real-time processing tab with all-pass filters and signal processing"""
        layout = QHBoxLayout()
        
        # Combine all-pass and real-time panels
        
        right_side = self.setup_signal_panel()
        layout.addWidget(right_side)
        
        self.real_time_tab.setLayout(layout)

    # Add tab styling
    def setup_tab_styling(self):
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ 
                border: 1px solid {ACCENT_COLOR};
                background: {DARK_PRIMARY};
            }}
            QTabBar::tab {{
                background: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                padding: 8px;
                margin: 2px;
            }}
            QTabBar::tab:selected {{
                background: {ACCENT_COLOR};
            }}
        """)
            
    def setup_dark_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(DARK_PRIMARY))
        palette.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.Base, QColor(DARK_SECONDARY))
        palette.setColor(QPalette.AlternateBase, QColor(DARK_PRIMARY))
        palette.setColor(QPalette.ToolTipBase, QColor(DARK_PRIMARY))
        palette.setColor(QPalette.ToolTipText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.Text, QColor(TEXT_COLOR))
        palette.setColor(QPalette.Button, QColor(DARK_SECONDARY))
        palette.setColor(QPalette.ButtonText, QColor(TEXT_COLOR))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(ACCENT_COLOR))
        palette.setColor(QPalette.Highlight, QColor(ACCENT_COLOR))
        palette.setColor(QPalette.HighlightedText, QColor(TEXT_COLOR))
        self.setPalette(palette)

    def initialize_plots(self):
        # Enhance z-plane
        self.z_ax = self.z_plane_figure.add_subplot(111, facecolor=PLOT_BG)
        
        # Enhanced unit circle
        circle = Circle((0, 0), 1, fill=False, color=PLOT_TEXT, linestyle='--', linewidth=2)
        self.z_ax.add_artist(circle)
        
        # Enhanced grid and labels
        self.z_ax.grid(True, color=PLOT_GRID, linestyle='--', alpha=0.5)
        self.z_ax.set_aspect('equal')
        self.z_ax.set_xlim(-2, 2)
        self.z_ax.set_ylim(-2, 2)
        
        # Add major and minor grid lines
        self.z_ax.grid(True, which='major', color=PLOT_GRID, linestyle='-', alpha=0.5)
        self.z_ax.grid(True, which='minor', color=PLOT_GRID, linestyle=':', alpha=0.3)
        self.z_ax.minorticks_on()
        
        # Enhanced labels
        self.z_ax.set_xlabel('Real Part', color=PLOT_TEXT, fontsize=12)
        self.z_ax.set_ylabel('Imaginary Part', color=PLOT_TEXT, fontsize=12)
        self.z_ax.set_title('Z-Plane Plot', color=PLOT_TEXT, fontsize=14, pad=20)
        
        # Add axes lines
        self.z_ax.axhline(y=0, color=PLOT_TEXT, linestyle='-', alpha=0.3)
        self.z_ax.axvline(x=0, color=PLOT_TEXT, linestyle='-', alpha=0.3)
        
        # Update frequency response plots styling
        for ax in [self.mag_ax, self.phase_ax]:
            ax.set_facecolor(PLOT_BG)
            ax.grid(True, which='both', color=PLOT_GRID, linestyle='--', alpha=0.5)
            ax.tick_params(colors=PLOT_TEXT, which='both')
            ax.minorticks_on()
            
            for spine in ax.spines.values():
                spine.set_color(PLOT_TEXT)
                spine.set_linewidth(1.5)


    def set_mode(self, mode):
        """Sets the current tool mode (zero/pole/drag) and updates button states"""
        # Deactivate current mode if clicking active mode
        if self.current_mode == mode:
            self.current_mode = None
            self.add_zero_btn.setChecked(False)
            self.add_pole_btn.setChecked(False)
        else:
            # Activate new mode
            self.current_mode = mode
            self.add_zero_btn.setChecked(mode == 'zero')
            self.add_pole_btn.setChecked(mode == 'pole')
            
        # Update cursor based on mode
        if self.current_mode:
            self.z_plane_canvas.setCursor(Qt.CrossCursor)
        else:
            self.z_plane_canvas.setCursor(Qt.ArrowCursor)


    def clear_all(self):
        self.zeros = []
        self.poles = []
        self.add_to_history()
        self.update_plots()
            
            
    def clear_zeros(self):
        self.zeros = []
        self.update_plots()

    def clear_poles(self):
        self.poles = []
        self.update_plots()

    
    def swap_zeros_poles(self):
        self.zeros, self.poles = self.poles.copy(), self.zeros.copy()
        self.add_to_history()
        self.update_plots()

    
    def update_plots(self):
        self.z_ax.clear()
        
        # Redraw enhanced unit circle and grid
        circle = Circle((0, 0), 1, fill=False, color=PLOT_TEXT, linestyle='--', linewidth=2)
        self.z_ax.add_artist(circle)
        
        # Enhanced grid setup
        self.z_ax.grid(True, which='both', color=PLOT_GRID, linestyle='--', alpha=0.5)
        self.z_ax.set_aspect('equal')
        self.z_ax.set_xlim(-2, 2)
        self.z_ax.set_ylim(-2, 2)
        
        # Plot zeros and poles with enhanced markers
        for zero in self.zeros:
            self.z_ax.plot(zero.real, zero.imag, 'o', color='blue', 
                        markersize=12, markeredgewidth=2, 
                        markerfacecolor='none', label='Zeros')
        
        for pole in self.poles:
            self.z_ax.plot(pole.real, pole.imag, 'x', color='red',
                        markersize=12, markeredgewidth=2,
                        label='Poles')
        
        # Enhanced axes and labels
        self.z_ax.set_xlabel('Real Part', color=PLOT_TEXT, fontsize=12)
        self.z_ax.set_ylabel('Imaginary Part', color=PLOT_TEXT, fontsize=12)
        self.z_ax.set_title('Z-Plane Plot', color=PLOT_TEXT, fontsize=14, pad=20)
        
        # Add legend
        handles, labels = self.z_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.z_ax.legend(by_label.values(), by_label.keys(), 
                        loc='upper right', facecolor=PLOT_BG, 
                        edgecolor=PLOT_TEXT)
        
        self.z_plane_canvas.draw()
        self.update_frequency_response()
        
    def update_frequency_response(self):
        """Update frequency response including all-pass effects"""
        # Calculate frequency points
        w = np.linspace(0, np.pi, 2000)
        z = np.exp(1j * w)
        
        # Main filter transfer function
        H = np.ones_like(z, dtype=complex)
        for zero in self.zeros:
            H *= (z - zero)
        for pole in self.poles:
            H /= (z - pole)
        
        # Apply all-pass filters
        if hasattr(self, 'all_pass_enabled') and self.all_pass_enabled.isChecked():
            for i in range(self.all_pass_list.count()):
                item = self.all_pass_list.item(i)
                if item.checkState() == Qt.Checked:
                    filter = self.all_pass_library.get_filter(i)
                    if filter:
                        # Calculate all-pass transfer function properly
                        num = z - filter.zero  # Numerator
                        den = 1 - filter.pole * z  # Denominator
                        H_ap = num / den
                        H *= H_ap  # Apply to main transfer function
        
        # Calculate magnitude and phase responses
        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.unwrap(np.angle(H)) * 180 / np.pi  # Convert to degrees
        
        # Clear previous plots
        self.mag_ax.clear()
        self.phase_ax.clear()
        
        # Plot magnitude response
        self.mag_ax.plot(w/np.pi, mag_db, 'w-', linewidth=2)
        self.mag_ax.set_ylabel('Magnitude (dB)', color=PLOT_TEXT)
        self.mag_ax.set_title('Magnitude Response', color=PLOT_TEXT)
        self.mag_ax.grid(True)
        
        # Plot phase response
        self.phase_ax.plot(w/np.pi, phase_deg, 'w-', linewidth=2)
        self.phase_ax.set_xlabel('Normalized Frequency (×π rad/sample)', color=PLOT_TEXT)
        self.phase_ax.set_ylabel('Phase (degrees)', color=PLOT_TEXT)
        self.phase_ax.set_title('Phase Response', color=PLOT_TEXT)
        self.phase_ax.grid(True)
        
        # Update plots
        self.freq_figure.tight_layout()
        self.freq_canvas.draw()

    def setup_toolbar(self):
        toolbar = QToolBar()
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # File operations
        save_action = QAction("Save Filter", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_filter)
        
        load_action = QAction("Load Filter", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_filter)
        
        # Edit operations
        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo)
        
        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self.redo)
        
        # Add actions to toolbar
        toolbar.addAction(save_action)
        toolbar.addAction(load_action)
        toolbar.addSeparator()
        toolbar.addAction(undo_action)
        toolbar.addAction(redo_action)
        
        # Filter library dropdown
        filter_combo = QComboBox()
        filter_combo.addItems([
            "Butterworth LPF",
            "Chebyshev LPF",
            "Elliptic LPF",
            "Butterworth HPF",
            "Chebyshev HPF",
            "Elliptic HPF",
            "Bessel LPF",
            "Bessel HPF",
            "Gaussian LPF",
            "Notch Filter"

        ])
        filter_combo.currentTextChanged.connect(self.load_preset_filter)
        toolbar.addWidget(filter_combo)

    # def load_preset_filter(self, filter_name):
    #     if filter_name == "Butterworth LPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     elif filter_name == "Chebyshev LPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     elif filter_name == "Elliptic LPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     elif filter_name == "Butterworth HPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     elif filter_name == "Chebyshev HPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     elif filter_name == "Elliptic HPF":
    #         self.zeros = []
    #         self.poles = []
    #         self.zeros = [complex(0, 1)] * 2
    #         self.poles = [complex(-1, 0)] * 2
    #     self.update_plots()

    def load_preset_filter(self, filter_name):
        """
        Load preset filters with predefined zeros and poles.
        """
        if filter_name == "Butterworth LPF":
            self.zeros = []  # No zeros for LPF
            self.poles = [
                complex(-0.7071, 0.7071),  # Pole for LPF
                complex(-0.7071, -0.7071)   # Pole for LPF
            ]  # 2nd-order Butterworth LPF

        elif filter_name == "Butterworth HPF":
            self.zeros = []  # No zeros for HPF
            self.poles = [
                complex(0.7071, 0.7071),   # Pole for HPF (mirrored)
                complex(0.7071, -0.7071)    # Pole for HPF (mirrored)
            ]  # 2nd-order Butterworth HPF

        elif filter_name == "Chebyshev LPF":
            self.zeros = []  # No zeros for the LPF
            self.poles = [
                complex(-0.5176, 0.8550),  # Pole for LPF
                complex(-0.5176, -0.8550)   # Pole for LPF
            ]  # 2nd-order Chebyshev LPF (0.5 dB ripple)

        elif filter_name == "Chebyshev HPF":
            self.zeros = []  # No zeros for the HPF
            self.poles = [
                complex(0.5176, 0.8550),   # Pole for HPF (mirrored)
                complex(0.5176, -0.8550)    # Pole for HPF (mirrored)
            ]  # 2nd-order Chebyshev HPF (0.5 dB ripple)

        elif filter_name == "Elliptic LPF":
            self.zeros = [
                complex(0, 0.9),
                complex(0, -0.9)
            ]  # Zeros for the LPF
            self.poles = [
                complex(-0.6986, 0.5375),
                complex(-0.6986, -0.5375)
            ]  # Poles for the LPF
        elif filter_name == "Elliptic HPF":
            self.zeros = [
                0, 0  # Zeros at the origin for HPF
            ]  # Zeros for the HPF
            self.poles = [
                complex(0.6986, 0.5375),  # Mirror the pole locations for HPF
                complex(0.6986, -0.5375)
            ]  # Poles for the HPF (mirrored to make it a high-pass filter)

        elif filter_name == "Bessel LPF":
            self.zeros = []  # No zeros for LPF
            self.poles = [
                complex(-0.866, 0.5),
                complex(-0.866, -0.5)
            ]  # Example: 2nd-order Bessel LPF poles
        elif filter_name == "Bessel HPF":
            self.zeros = [
                0, 0  # Two zeros at the origin for a 2nd-order HPF
            ]
            self.poles = [
                complex(-0.866, 0.5),
                complex(-0.866, -0.5)
            ]  # Poles remain the same, but zeros are added to invert the response

        elif filter_name == "Gaussian LPF":
            self.zeros = []  # No zeros for LPF
            self.poles = [
                complex(-0.707, 0)  # Example of a simple Gaussian pole for LPF
                # More poles can be added for higher-order Gaussian filters
            ]  # 2nd-order Gaussian LPF
        elif filter_name == "Notch Filter":
            self.zeros = [
                complex(1, 0),
                complex(-1, 0)
            ]
            self.poles = [
                complex(0.95, 0.1),
                complex(0.95, -0.1)
            ]
        else:
            print(f"Unknown filter: {filter_name}")
            return

        # Update the plots with the newly loaded zeros and poles
        self.update_plots()


    
    def setup_plots(self):
        # Add navigation toolbar
        self.z_toolbar = NavigationToolbar(self.z_plane_canvas, self)
        self.freq_toolbar = NavigationToolbar(self.freq_canvas, self)
        
        # Add coordinate display
        self.z_plane_canvas.mpl_connect('motion_notify_event', self.update_coords)
        self.coord_label = QLabel()
        self.statusBar().addWidget(self.coord_label)
        
    def update_coords(self, event):
        if event.inaxes:
            self.coord_label.setText(f'x={event.xdata:.2f}, y={event.ydata:.2f}')
            
    def save_filter(self):
        """Save filter coefficients to JSON file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Filter", 
            "", 
            "Filter Files (*.flt)"
        )
        
        if filename:
            if not filename.endswith('.flt'):
                filename += '.flt'
                
            # Convert complex numbers to [real, imag] pairs for JSON
            data = {
                'zeros': [[z.real, z.imag] for z in self.zeros],
                'poles': [[p.real, p.imag] for p in self.poles]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

    def load_filter(self):
        """Load filter coefficients from JSON file"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Filter", 
            "", 
            "Filter Files (*.flt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
                # Convert [real, imag] pairs back to complex numbers
                self.zeros = [complex(z[0], z[1]) for z in data['zeros']]
                self.poles = [complex(p[0], p[1]) for p in data['poles']]
                
                self.add_to_history()
                self.update_plots()
                
            except Exception as e:
                QMessageBox.warning(self, "Load Error", 
                                f"Error loading filter file: {str(e)}")
            
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.zeros = state['zeros'].copy()
            self.poles = state['poles'].copy()
            self.all_pass_filters = [AllPassFilter(a) for a in state['all_pass']]
            self.update_plots()
            
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.zeros = state['zeros'].copy()
            self.poles = state['poles'].copy()
            self.all_pass_filters = [AllPassFilter(a) for a in state['all_pass']]
            self.update_plots()

    def save_state(self):
        """Save current filter state for undo/redo"""
        state = {
            'zeros': self.zeros.copy(),
            'poles': self.poles.copy(),
            'all_pass': self.all_pass_filters.copy()
        }
        self.history.append(state)
        self.current_state += 1

    def reset_filter_states(self):
        """Reset filter states when coefficients change"""
        self.direct_state = None
        self.cascade_state = None
        self.input_signal.clear()
        self.output_signal.clear()

    def find_conjugate(self, idx, points):
        """Find index of conjugate pair for given point"""
        point = points[idx]
        for i, p in enumerate(points):
            # Check if this is a different point with conjugate coordinates
            if i != idx and abs(p.real - point.real) < 0.01 and abs(p.imag + point.imag) < 0.01:
                return i
        return None

    def on_press(self, event):
        if event.inaxes != self.z_ax:
            return

        x, y = event.xdata, event.ydata
            
        # Right click to delete
        if event.button == 3:  # Right click
            self.handle_deletion(x, y)
            return
        
        # Left click to add or start dragging
        elif event.button == 1:
            # Check for dragging first
            for i, zero in enumerate(self.zeros):
                if abs(zero.real - x) < 0.1 and abs(zero.imag - y) < 0.1:
                    self.dragging = True
                    self.drag_target = i
                    self.drag_type = 'zero'
                    self.z_plane_canvas.setCursor(Qt.ClosedHandCursor)
                    return
                    
            for i, pole in enumerate(self.poles):
                if abs(pole.real - x) < 0.1 and abs(pole.imag - y) < 0.1:
                    self.dragging = True
                    self.drag_target = i
                    self.drag_type = 'pole'
                    self.z_plane_canvas.setCursor(Qt.ClosedHandCursor)
                    return
            
            # If not dragging, add new point if in add mode
            if self.current_mode:
                self.add_new_point(x, y)

    def add_new_point(self, x, y):
        """Add new point with conjugate pair tracking"""
        if self.current_mode == 'pole':
            radius = np.sqrt(x*x + y*y)
            if radius >= 1.0:
                QMessageBox.warning(self, "Unstable Filter", 
                                "Poles must be inside the unit circle")
                return
                
        points = self.zeros if self.current_mode == 'zero' else self.poles
        pairs_dict = self.conjugate_pairs['zeros' if self.current_mode == 'zero' else 'poles']
        
        # Add main point
        idx = len(points)
        points.append(complex(x, y))
        
        # Add conjugate if enabled
        if self.conjugate_check.isChecked():
            conj_idx = len(points)
            points.append(complex(x, -y))
            # Store bidirectional reference
            pairs_dict[idx] = conj_idx
            pairs_dict[conj_idx] = idx
        
        self.add_to_history()
        self.update_plots()

    def on_motion(self, event):
        """Handle dragging with conjugate pair movement"""
        if not self.dragging or event.inaxes != self.z_ax:
            return
                
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
                
        # Check stability for poles
        if self.drag_type == 'pole':
            radius = np.sqrt(x*x + y*y)
            if radius >= 1.0:
                return
                
        # Get active arrays
        points = self.zeros if self.drag_type == 'zero' else self.poles
        pairs_dict = self.conjugate_pairs['zeros' if self.drag_type == 'zero' else 'poles']
        
        # Update main point
        points[self.drag_target] = complex(x, y)
        
        # Update conjugate if it exists
        if self.drag_target in pairs_dict:
            conj_idx = pairs_dict[self.drag_target]
            points[conj_idx] = complex(x, -y)  # Mirror y-coordinate only
        
        self.update_plots()

    def handle_deletion(self, x, y):
        """Handle deletion of points and their conjugates"""
        # Check zeros
        for i, zero in enumerate(self.zeros):
            if abs(zero.real - x) < 0.1 and abs(zero.imag - y) < 0.1:
                # Delete conjugate if exists
                if i in self.conjugate_pairs['zeros']:
                    conj_idx = self.conjugate_pairs['zeros'][i]
                    # Remove the one with larger index first
                    if conj_idx > i:
                        self.zeros.pop(conj_idx)
                        self.zeros.pop(i)
                    else:
                        self.zeros.pop(i)
                        self.zeros.pop(conj_idx)
                    # Clean up conjugate pairs
                    del self.conjugate_pairs['zeros'][i]
                    del self.conjugate_pairs['zeros'][conj_idx]
                else:
                    self.zeros.pop(i)
                self.add_to_history()
                self.update_plots()
                return

        # Check poles - same logic as zeros
        for i, pole in enumerate(self.poles):
            if abs(pole.real - x) < 0.1 and abs(pole.imag - y) < 0.1:
                if i in self.conjugate_pairs['poles']:
                    conj_idx = self.conjugate_pairs['poles'][i]
                    if conj_idx > i:
                        self.poles.pop(conj_idx)
                        self.poles.pop(i)
                    else:
                        self.poles.pop(i)
                        self.poles.pop(conj_idx)
                    del self.conjugate_pairs['poles'][i]
                    del self.conjugate_pairs['poles'][conj_idx]
                else:
                    self.poles.pop(i)
                self.add_to_history()
                self.update_plots()
                return

    def on_release(self, event):
        """Handle mouse release after dragging"""
        if self.dragging:
            self.dragging = False
            self.add_to_history()
            self.drag_target = None
            self.drag_type = None
            # Restore drag mode cursor
            if self.current_mode == 'drag':
                self.z_plane_canvas.setCursor(Qt.OpenHandCursor)

    

    def generate_c_code(self):
        """Generate C code for the current filter design"""
        try:
            # Get coefficients based on selected form
            if self.direct_form.isChecked():
                coeffs = self.generate_direct_form_II()
                c_code = self._generate_direct_form_c(coeffs)
            else:
                coeffs = self.generate_cascade_form()
                c_code = self._generate_cascade_form_c(coeffs)
            
            # Save to file with proper path handling
            file_name = "filter_implementation.c"
            with open(file_name, "w") as file:
                file.write(c_code)
            print(f"C code saved to {file_name}")
        except Exception as e:
            print(f"Error generating C code: {e}")

    def _generate_direct_form_c(self, coeffs):
        """Generate Direct Form II implementation"""
        b = coeffs['b']
        a = coeffs['a']
        
        template = [
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <math.h>',
            '',
            f'#define NUM_B {len(b)}',
            f'#define NUM_A {len(a)}',
            '',
            'typedef struct {',
            '    double *state;',
            '    int state_size;',
            '} FilterState;',
            '',
            f'static const double b[NUM_B] = {{{", ".join([f"{x.real:.10f}" for x in b])}}};\n',
            f'static const double a[NUM_A] = {{{", ".join([f"{x.real:.10f}" for x in a])}}};\n',
            '',
            'FilterState* filter_init(void) {',
            '    FilterState* f = (FilterState*)malloc(sizeof(FilterState));',
            '    f->state_size = NUM_A - 1;',
            '    f->state = (double*)calloc(f->state_size, sizeof(double));',
            '    return f;',
            '}',
            '',
            'double filter_process(FilterState* f, double x) {',
            '    double w = x;',
            '    for(int i = 0; i < f->state_size; i++) {',
            '        w -= a[i+1] * f->state[i];',
            '    }',
            '    double y = b[0] * w;',
            '    for(int i = 0; i < f->state_size; i++) {',
            '        y += b[i+1] * f->state[i];',
            '    }',
            '    for(int i = f->state_size-1; i > 0; i--) {',
            '        f->state[i] = f->state[i-1];',
            '    }',
            '    f->state[0] = w;',
            '    return y;',
            '}',
            '',
            'void filter_free(FilterState* f) {',
            '    free(f->state);',
            '    free(f);',
            '}',
            '',
            'int main(void) {',
            '    FilterState* filter = filter_init();',
            '    double input[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};',
            '    double output;',
            '    for(int i = 0; i < 10; i++) {',
            '        output = filter_process(filter, input[i]);',
            '        printf("Sample %d: in=%.4f, out=%.4f\\n", i, input[i], output);',
            '    }',
            '    filter_free(filter);',
            '    return 0;',
            '}'
        ]
        
        return '\n'.join(template)

    def _generate_cascade_form_c(self, coeffs):
        """Generate Cascade Form implementation"""
        template = [
            '#include <stdio.h>',
            '#include <stdlib.h>',
            '#include <math.h>',
            '',
            f'#define NUM_SECTIONS {len(coeffs)}',
            '',
            'typedef struct {',
            '    double state[NUM_SECTIONS][2];',
            '} FilterState;',
            '',
            'static const double sos[NUM_SECTIONS][6] = {',
            ',\n'.join([f'    {{ {", ".join(f"{x:.10f}" for x in section)} }}' for section in coeffs]),
            '};',
            '',
            'FilterState* filter_init(void) {',
            '    FilterState* f = (FilterState*)malloc(sizeof(FilterState));',
            '    for(int i = 0; i < NUM_SECTIONS; i++) {',
            '        f->state[i][0] = 0.0;',
            '        f->state[i][1] = 0.0;',
            '    }',
            '    return f;',
            '}',
            '',
            'double filter_process(FilterState* f, double x) {',
            '    double y = x, w;',
            '    for(int i = 0; i < NUM_SECTIONS; i++) {',
            '        w = y - sos[i][4]*f->state[i][0] - sos[i][5]*f->state[i][1];',
            '        y = sos[i][0]*w + sos[i][1]*f->state[i][0] + sos[i][2]*f->state[i][1];',
            '        f->state[i][1] = f->state[i][0];',
            '        f->state[i][0] = w;',
            '    }',
            '    return y;',
            '}',
            '',
            'void filter_free(FilterState* f) {',
            '    free(f);',
            '}',
            '',
            'int main(void) {',
            '    FilterState* filter = filter_init();',
            '    double input[10] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};',
            '    double output;',
            '    for(int i = 0; i < 10; i++) {',
            '        output = filter_process(filter, input[i]);',
            '        printf("Sample %d: in=%.4f, out=%.4f\\n", i, input[i], output);',
            '    }',
            '    filter_free(filter);',
            '    return 0;',
            '}'
        ]
        
        return '\n'.join(template)


    ########################## real time plotting ############################

    def update_speed(self, value):
        """Update processing speed with proper timing"""
        self.processing_speed = value
        interval = max(1, int(1000 / value))  # Ensure minimum 1ms interval
        self.process_timer.setInterval(interval)
        
        # Clear buffers on speed change
        if len(self.input_signal) > value * 10:
            self.input_signal = deque(list(self.input_signal)[-value*10:], 
                                    maxlen=value*10)
            self.output_signal = deque(list(self.output_signal)[-value*10:], 
                                    maxlen=value*10)
    
    def process_next_sample(self):
        """Main processing chain"""
        if not self.input_signal:
            return
                    
        try:
            # 1. Get input sample
            x = self.input_signal[-1]
            
            # 2. Apply main filter (from z-plane design)
            y = self.apply_selected_filter(x)
            
            # 3. Apply all-pass filters (optional)
            if self.all_pass_enabled.isChecked():
                y = self.apply_all_pass_filters(y)
                    
            # 4. Store output and ensure signal buffers match
            self.output_signal.append(float(y))
            
            # Ensure input and output buffers have same length
            while len(self.output_signal) < len(self.input_signal):
                self.output_signal.append(float(y))
            while len(self.input_signal) < len(self.output_signal):
                self.input_signal.append(float(x))
                
            # 5. Update visualization
            self.update_signal_plots()
        
        except Exception as e:
            print(f"Error processing sample: {e}")


    def on_mouse_draw(self, event):
        """Handle mouse movement in drawing area to generate input signal"""
        if not hasattr(self, 'last_y'):
            self.last_y = event.y()
            return
                
        # Calculate vertical displacement for frequency
        dy = event.y() - self.last_y
        
        # Convert mouse movement to signal value (-1 to 1 range)
        y = (self.draw_area.height() - event.y()) / self.draw_area.height() * 2 - 1
        
        # Add to input buffer with rate limiting
        if len(self.input_signal) < 10000:  # Maintain max buffer size
            self.input_signal.append(y)
        else:
            self.input_signal = self.input_signal[1:] + [y]
        
        self.last_y = event.y()
        
    def process_signal(self):
        # Implement actual filter processing using difference equation
        x = self.input_signal[-1]
        
        # Apply selected filter (Direct Form II or Cascade)
        y = self.apply_selected_filter(x)
        
        # Apply selected all-pass filters
        if self.all_pass_enabled:
            y = self.apply_all_pass_filters(y)
            
        self.output_signal.append(y)

    def apply_selected_filter(self, x):
        """Apply filter with normalization"""
        try:
            if len(self.zeros) == 0 and len(self.poles) == 0:
                return x
                
            if self.direct_form.isChecked():
                coeffs = self.generate_direct_form_II()
                y = self.apply_direct_form(x, coeffs)
            else:
                coeffs = self.generate_cascade_form()
                y = self.apply_cascade_form(x, coeffs)
                
            # Normalize output to prevent overflow
            if abs(y) > 1.0:
                y = y / abs(y)
                
            return float(y)
                    
        except Exception as e:
            print(f"Error applying filter: {e}")
            return x

    def apply_direct_form(self, x, coeffs):
        try:
            # Convert coefficients to complex arrays
            b = np.array(coeffs['b'], dtype=complex)
            a = np.array(coeffs['a'], dtype=complex)
            
            state_size = max(len(b), len(a)) - 1
            if self.direct_state is None or len(self.direct_state) != state_size:
                self.direct_state = np.zeros(state_size, dtype=complex)
                
            # Direct Form II implementation
            w = complex(x)  # Input to state
            for i in range(1, len(a)):
                w = w - a[i] * self.direct_state[i-1]
                
            y = b[0] * w  # Output computation
            for i in range(1, len(b)):
                y = y + b[i] * self.direct_state[i-1]
                
            # Update state correctly
            self.direct_state = np.roll(self.direct_state, 1)
            self.direct_state[0] = w
            
            # Return real part for output
            return float(np.real(y))

        except Exception as e:
            print(f"Error applying Direct Form II: {e}")
            return x

    def generate_direct_form_II(self):
        """Convert zeros and poles to direct form II coefficients"""
        try:
            # Handle empty filter case
            if not self.zeros and not self.poles:
                return {'b': [1.0], 'a': [1.0]}
                
            # Convert complex zeros/poles to polynomial coefficients
            b = np.poly(self.zeros) if self.zeros else np.array([1.0])
            a = np.poly(self.poles) if self.poles else np.array([1.0])
            
            # Keep as complex numbers
            b = np.array(b, dtype=complex)
            a = np.array(a, dtype=complex)
            
            # Normalize coefficients
            if len(a) > 0:
                b = b / a[0]
                a = a / a[0]
            
            return {
                'b': b.tolist(),
                'a': a.tolist()
            }
            
        except Exception as e:
            print(f"Error generating coefficients: {e}")
            return {'b': [1.0], 'a': [1.0]}

    def apply_cascade_form(self, x, coeffs):
        """Apply Cascade Form implementation"""
        # Initialize state properly for each second-order section
        if self.cascade_state is None or self.cascade_state.shape[0] != len(coeffs):
            self.cascade_state = np.zeros((len(coeffs), 2))
            
        y = x
        for i, section in enumerate(coeffs):
            # Correct implementation of second-order section
            w0 = y - section[4]*self.cascade_state[i,0] - section[5]*self.cascade_state[i,1]
            y = section[0]*w0 + section[1]*self.cascade_state[i,0] + section[2]*self.cascade_state[i,1]
            
            # Update states correctly
            self.cascade_state[i,1] = self.cascade_state[i,0]
            self.cascade_state[i,0] = w0
            
        return y

    def apply_all_pass_filters(self, x):
        """Process input through all enabled all-pass filters"""
        y = x
        try:
            if self.all_pass_enabled.isChecked():
                for i in range(self.all_pass_list.count()):
                    item = self.all_pass_list.item(i)
                    if item.checkState() == Qt.Checked:
                        filter = self.all_pass_library.get_filter(i)
                        if filter:
                            y = filter.process(y)
        except Exception as e:
            print(f"Error in all-pass filtering: {e}")
            
        return y
            
    def update_signal_plots(self):
        """Update scrolling signal display"""
        if not self.input_signal or not self.output_signal:
            return
                
        # Get window size
        window = self.window_spin.value()
        
        # Get recent samples and ensure same length
        input_data = np.array(list(self.input_signal)[-window:])
        output_data = np.array(list(self.output_signal)[-window:])
        
        # Ensure both arrays have same length
        min_len = min(len(input_data), len(output_data))
        if min_len == 0:
            return
            
        input_data = input_data[-min_len:]
        output_data = output_data[-min_len:]
        
        # Create time axis in seconds
        dt = 1.0 / self.processing_speed
        t = np.arange(min_len) * dt
        
        # Update plots
        self.input_curve.setData(t, input_data)
        self.output_curve.setData(t, output_data)


    def export_filter(self):
        """Export filter realization diagram"""
        try:
            # Create new figure for block diagram
            fig = plt.figure(figsize=(12, 8))
            
            if self.direct_form.isChecked():
                self._draw_direct_form(fig)
                title = "Direct Form II Implementation"
            else:
                self._draw_cascade_form(fig)
                title = "Cascade Form Implementation"
                
            plt.suptitle(title, fontsize=16, color='white')
            fig.patch.set_facecolor(DARK_PRIMARY)
            
            # Save dialog
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Filter Realization",
                "",
                "PNG Files (*.png);;All Files (*)"
            )
            
            if filename:
                if not filename.endswith('.png'):
                    filename += '.png'
                plt.savefig(filename, facecolor=DARK_PRIMARY, bbox_inches='tight')
                plt.close(fig)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Filter realization saved to {filename}"
                )
                
        except Exception as e:
            QMessageBox.warning(
                self,
                "Export Error",
                f"Error exporting filter realization: {str(e)}"
            )

    def _draw_direct_form(self, fig):
        """Draw Direct Form II block diagram"""
        coeffs = self.generate_direct_form_II()
        b = np.array(coeffs['b'])
        a = np.array(coeffs['a'])
        
        # Create subplot
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_SECONDARY)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Match array lengths
        if len(b) < len(a):
            b = np.pad(b, (0, len(a) - len(b)))
        elif len(a) < len(b):
            a = np.pad(a, (0, len(b) - len(a)))
        
        order = len(b) - 1
        
        # Draw delay elements and arrows
        for i in range(order):
            # Input feedforward
            if a[i] != 0:
                ax.arrow(0.6, 0.7 - (i * 0.4), -0.18, 0, head_width=0.02, 
                        head_length=0.02, color='white')
                ax.text(0.45, 0.72 - (i * 0.4), f"{a[i].real:.2f}", 
                    fontsize=12, color=ACCENT_COLOR)
            
            # Output feedback
            if b[i] != 0:
                ax.arrow(0.6, 0.7 - (i * 0.4), 0.38, 0, head_width=0.02,
                        head_length=0.02, color='white')
                ax.text(0.8, 0.72 - (i * 0.4), f"{b[i].real:.2f}",
                    fontsize=12, color=ACCENT_COLOR)
            
            # Vertical connections
            ax.arrow(0.6, 0.7 - (i * 0.4), 0, -0.2, head_width=0.02,
                    head_length=0.02, color='white')
            
            # Delay elements
            ax.text(0.6, 0.5 - (i * 0.4), r"$Z^{-1}$", fontsize=9,
                    ha="center", va="center",
                    bbox=dict(boxstyle="square", facecolor=ACCENT_COLOR))
            
            # Continue vertical connections
            ax.arrow(0.6, 0.4 - (i * 0.4), 0, -0.08, head_width=0.02,
                    head_length=0.02, color='white')
        
        # Final coefficients
        if a[order] != 0:
            ax.arrow(0.6, 0.7 - (order * 0.4), -0.18, 0, head_width=0.02,
                    head_length=0.02, color='white')
            ax.text(0.45, 0.72 - (order * 0.4), f"{a[order].real:.2f}",
                    fontsize=12, color=ACCENT_COLOR)
        
        if b[order] != 0:
            ax.arrow(0.6, 0.7 - (order * 0.4), 0.38, 0, head_width=0.02,
                    head_length=0.02, color='white')
            ax.text(0.8, 0.72 - (order * 0.4), f"{b[order].real:.2f}",
                    fontsize=12, color=ACCENT_COLOR)
        
        # Draw summation nodes
        for i in range(order):
            if a[i + 1] != 0:
                ax.text(0.38, 0.7 - (i * 0.4), "+", fontsize=10,
                    bbox=dict(boxstyle="circle", facecolor=ACCENT_COLOR))
                ax.arrow(0.4, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02,
                        head_length=0.02, color='white')
            
            if b[i + 1] != 0:
                ax.text(0.99, 0.7 - (i * 0.4), "+", fontsize=10,
                    bbox=dict(boxstyle="circle", facecolor=ACCENT_COLOR))
                ax.arrow(1, 0.32 - (i * 0.4), 0, 0.3, head_width=0.02,
                        head_length=0.02, color='white')
        
        # Input/Output labels
        ax.arrow(0.25, 0.7, 0.1, 0, head_width=0.02, head_length=0.02, color='white')
        ax.arrow(1.24, 0.7, -0.18, 0, head_width=0.02, head_length=0.02, color='white')
        ax.text(0.2, 0.7, "x[n]", fontsize=12, ha="center", color='white',
                bbox=dict(boxstyle="round", facecolor=ACCENT_COLOR))
        ax.text(1.26, 0.7, "y[n]", fontsize=12, ha="center", color='white',
                bbox=dict(boxstyle="round", facecolor=ACCENT_COLOR))
        
        # Set plot limits
        ax.set_xlim(-0.1, 1.3)
        ax.set_ylim(-1, 1)

    def _draw_cascade_form(self, fig):
        """Draw Cascade Form block diagram"""
        coeffs = self.generate_cascade_form()
        
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_SECONDARY)
        ax.set_xticks([])
        ax.set_yticks([])

        x_offset = 0.1
        width = 0.2
        spacing = 0.25
        
        # Draw sections
        for i, section in enumerate(coeffs):
            # Section box
            rect = plt.Rectangle(
                (x_offset + i*spacing, 0.3),
                width, 0.4,
                facecolor=ACCENT_COLOR,
                alpha=0.5
            )
            ax.add_patch(rect)
            
            # Fix transfer function formatting
            num = f"{section[0]:.2f} + {section[1]:.2f}z^{{-1}} + {section[2]:.2f}z^{{-2}}"
            den = f"{1:.2f} + {section[4]:.2f}z^{{-1}} + {section[5]:.2f}z^{{-2}}"
            
            ax.text(
                x_offset + i*spacing + width/2,
                0.5,
                f"$\\frac{{{num}}}{{{den}}}$",  # Proper LaTeX fraction
                ha='center',
                va='center',
                color='white',
                fontsize=8
            )
            
            # Connection arrows
            if i < len(coeffs) - 1:
                ax.arrow(
                    x_offset + i*spacing + width,
                    0.5,
                    spacing - width,
                    0,
                    head_width=0.02,
                    color='white'
                )
        
        # Input/Output arrows and labels 
        ax.arrow(0, 0.5, x_offset-0.05, 0, head_width=0.02, color='white')
        ax.arrow(x_offset + len(coeffs)*spacing, 0.5, 0.1, 0, head_width=0.02, color='white')
        
        ax.text(-0.05, 0.5, "x[n]", ha='right', va='center', color='white',
                bbox=dict(boxstyle="round", facecolor=ACCENT_COLOR))
        ax.text(x_offset + len(coeffs)*spacing + 0.15, 0.5, "y[n]", 
                ha='left', va='center', color='white',
                bbox=dict(boxstyle="round", facecolor=ACCENT_COLOR))
        
        ax.set_xlim(-0.1, x_offset + len(coeffs)*spacing + 0.2)
        ax.set_ylim(0, 1)

    def generate_cascade_form(self):
        """Convert zeros and poles to cascade form coefficients"""
        try:
            # Get transfer function coefficients
            b = np.poly(self.zeros)
            a = np.poly(self.poles)
            
            # Convert to second-order sections
            sos = tf2sos(b, a, pairing='nearest')
            return sos.tolist()
        except Exception as e:
            print(f"Error generating cascade form: {e}")
            return np.array([[1, 0, 0, 1, 0, 0]]).tolist()

    
    def add_to_history(self):
        # Remove any redo states
        while len(self.history) > self.history_index + 1:
            self.history.pop()
        
        # Add current state
        state = {
            'zeros': self.zeros.copy(),
            'poles': self.poles.copy(),
            'all_pass': [f.a for f in self.all_pass_filters]
        }
        self.history.append(state)
        self.history_index += 1

    def setup_all_pass_panel(self):
        """Add panel for all-pass filter configuration and library"""
        panel = QGroupBox("All-Pass Filters")
        layout = QVBoxLayout()
        
        # Enable/disable all-pass filters
        self.all_pass_enabled = QCheckBox("Enable All-Pass Filters")
        self.all_pass_enabled.setStyleSheet(f"color: {TEXT_COLOR};")
        self.all_pass_enabled.stateChanged.connect(self.on_all_pass_enabled)
        
        # Library list with checkable items
        self.all_pass_list = QListWidget()
        self.all_pass_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        self.all_pass_list.itemChanged.connect(self.update_frequency_response)
        self.all_pass_enabled.stateChanged.connect(self.update_frequency_response)
        # Add default filters to list
        for name in self.all_pass_library.get_filter_names():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.all_pass_list.addItem(item)
        
        # Custom filter input
        custom_layout = QHBoxLayout()
        self.a_input = QLineEdit()
        self.a_input.setPlaceholderText("Enter coefficient (0-1)")
        self.a_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        
        add_btn = QPushButton("Add Custom Filter")
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        add_btn.clicked.connect(self.add_custom_filter)
        
        custom_layout.addWidget(self.a_input)
        custom_layout.addWidget(add_btn)
        
        # Add widgets to layout
        layout.addWidget(self.all_pass_enabled)
        layout.addWidget(self.all_pass_list)
        layout.addLayout(custom_layout)
        
        panel.setLayout(layout)
        return panel
    
    def on_all_pass_changed(self, item):
        """Handle all-pass filter enable/disable"""
        if self.all_pass_enabled.isChecked():
            self.update_frequency_response()


    def on_all_pass_enabled(self, state):
        """Handle enabling/disabling all-pass filters"""
        self.all_pass_list.setEnabled(state == Qt.Checked)
        self.a_input.setEnabled(state == Qt.Checked)
        # Update frequency response when enabling/disabling
        self.update_frequency_response()

    def add_custom_filter(self):
        """Add custom all-pass filter from input"""
        try:
            a = float(self.a_input.text())
            if 0 <= a <= 1:
                self.all_pass_library.add_filter(a)
                item = QListWidgetItem(f"a={a}")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                self.all_pass_list.addItem(item)
                self.a_input.clear()
                # Update frequency response when adding filter
                if self.all_pass_enabled.isChecked():
                    self.update_frequency_response()
            else:
                QMessageBox.warning(self, "Invalid Input", 
                                "Coefficient must be between 0 and 1")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                            "Please enter a valid number")
    
    def setup_signal_panel(self):
        """Setup real-time signal processing panel"""
        panel = QGroupBox("Real-time Processing")
        layout = QVBoxLayout()

        # Speed control with finer granularity
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)  # 1-100 points/sec
        self.speed_slider.setValue(10)  # Default 10 pts/sec
        
        self.speed_label = QLabel("10 pts/sec")
        self.speed_slider.valueChanged.connect(self.update_processing_speed)
        
        speed_layout.addWidget(QLabel("Processing Speed:"))
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_label)

        # Window size control
        window_layout = QHBoxLayout()
        self.window_spin = QSpinBox()
        self.window_spin.setRange(100, 1000)
        self.window_spin.setValue(200)
        self.window_spin.setSingleStep(50)
        window_layout.addWidget(QLabel("View Window (pts):"))
        window_layout.addWidget(self.window_spin)

        # Drawing area with coordinate display
        self.draw_area = QWidget()
        self.draw_area.setMinimumSize(300, 100)
        self.draw_area.setStyleSheet(f"""
            QWidget {{
                background-color: {DARK_SECONDARY};
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        self.draw_area.setMouseTracking(True)
        self.draw_area.installEventFilter(self)
        
        # Signal plots
        self.input_plot = pg.PlotWidget(title="Input Signal")
        self.output_plot = pg.PlotWidget(title="Filtered Signal") 
        
        for plot in [self.input_plot, self.output_plot]:
            plot.setBackground(PLOT_BG)
            plot.showGrid(x=True, y=True)
            plot.setLabel('bottom', "Time (s)")
            plot.setLabel('left', "Amplitude")
            plot.setYRange(-1.1, 1.1)
        
        # Link X and Y axes between plots
        self.output_plot.setXLink(self.input_plot)
        self.output_plot.setYLink(self.input_plot)
        
        # Enable mouse interaction
        for plot in [self.input_plot, self.output_plot]:
            plot.getViewBox().enableAutoRange(axis='x')
            plot.getViewBox().enableAutoRange(axis='y')
            plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)
            
        # Add curves with synchronized updates
        self.input_curve = self.input_plot.plot(pen='y')
        self.output_curve = self.output_plot.plot(pen='c')
        
        # Add widgets to layout
        layout.addLayout(speed_layout)
        layout.addLayout(window_layout)
        layout.addWidget(self.draw_area)
        layout.addWidget(self.input_plot)
        layout.addWidget(self.output_plot)
        
        panel.setLayout(layout)
        return panel

    def change_signal_type(self, signal_type):
        """Change input signal generation method"""
        self.signal_type = signal_type
        self.input_signal.clear()
        self.output_signal.clear()
        
        if signal_type != "Draw Input":
            # Start automated signal generation
            self.signal_timer = QTimer()
            self.signal_timer.timeout.connect(self.generate_signal)
            self.signal_timer.start(20)
        else:
            if hasattr(self, 'signal_timer'):
                self.signal_timer.stop()

    def generate_signal(self):
        """Generate selected signal type"""
        t = len(self.input_signal) * 0.02  # Time based on sample count
        
        if self.signal_type == "Sine Wave":
            y = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine wave
        elif self.signal_type == "Square Wave":
            y = np.sign(np.sin(2 * np.pi * 0.5 * t))  # 0.5 Hz square wave
        elif self.signal_type == "Noise":
            y = np.random.uniform(-1, 1)
        else:
            return
            
        self.input_signal.append(float(y))
        self.process_next_sample()

    def eventFilter(self, obj, event):
        """Handle mouse events in drawing area"""
        if obj is self.draw_area:
            if event.type() == event.MouseMove:
                self.handle_mouse_draw(event)
                return True
        return super().eventFilter(obj, event)

    def update_processing_speed(self, value):
        """Update processing speed and display"""
        self.processing_speed = value
        self.speed_label.setText(f"{value} pts/sec")
        
        # Update timer interval (ms)
        interval = int(1000 / value)
        self.process_timer.setInterval(interval)
        
        # Clear old data
        self.reset_signal_buffers()

    def handle_mouse_draw(self, event):
        """Generate input signal from mouse movement"""
        if not hasattr(self, 'last_pos'):
            self.last_pos = event.pos()
            self.last_time = time.time()
            return
            
        # Calculate mouse velocity 
        dt = time.time() - self.last_time
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()
        velocity = np.sqrt(dx*dx + dy*dy) / dt
        
        # Generate signal based on y position
        y = 1.0 - (2.0 * event.pos().y() / self.draw_area.height())
        
        # Add frequency component based on velocity
        if velocity > 0:
            freq = min(20, velocity / 100)  # Cap max frequency
            y *= np.sin(2 * np.pi * freq * dt)
        
        self.input_signal.append(float(y))
        
        # Update state
        self.last_pos = event.pos()
        self.last_time = time.time()
        
        self.process_next_sample()
    
    def process_all_pass(self, x):
        """Apply enabled all-pass filters to input sample"""
        y = x
        # Check if checkbox exists and is enabled
        if not hasattr(self, 'all_pass_enabled') or not self.all_pass_enabled.isChecked():
            return y
            
        # Loop through enabled filters in list
        for i in range(self.all_pass_list.count()):
            item = self.all_pass_list.item(i)
            if item and item.checkState() == Qt.Checked:
                filter = self.all_pass_library.get_filter(i)
                if filter:
                    y = filter.process(y)
                    
        return y
    
    def update_visualization(self):
        """Update signal visualization with new window size"""
        self.update_signal_plots()


class AllPassFilter:
    """
    All-pass filter implementation with unity magnitude response.
    
    Properties:
    - Magnitude response = 1 at all frequencies
    - Phase response varies with frequency
    - Used for phase correction without affecting magnitude
    
    Attributes:
        a (float): Filter coefficient (0 < a < 1)
        zero (float): Zero location (outside unit circle)
        pole (float): Pole location (inside unit circle)
        state (float): Filter state for real-time processing
    """
    def __init__(self, a):
        self.a = float(a)  # Coefficient
        self.zero = 1/self.a  # Reciprocal location (outside unit circle)
        self.pole = self.a   # Inside unit circle
        self.state = 0.0
        
    def process(self, x):
        """Process one sample through all-pass filter"""
        try:
            # Direct Form I implementation
            w = x - self.pole * self.state
            y = self.zero * w + self.state
            self.state = w  # Update state
            return y
        except Exception as e:
            print(f"Error in filter: {e}")
            return x

class AllPassLibrary:
    """
    Library of common all-pass filter configurations.
    
    Features:
    - Predefined coefficients for common phase corrections
    - Dynamic addition of custom filters
    - Named filter access
    
    Default coefficients:
    - 0.5: 90° phase shift at π/3
    - 0.7: 90° phase shift at π/2
    - 0.9: 90° phase shift at 2π/3
    - 0.95: Sharper phase transition
    - 0.98: Very sharp phase transition
    """

    def __init__(self):
        self.filters = []
        self.initialize_library()
    
    def initialize_library(self):
        """Initialize with common all-pass filter coefficients"""
        default_coeffs = [0.5, 0.7, 0.9, 0.95, 0.98]
        for a in default_coeffs:
            self.filters.append(AllPassFilter(a))
            
    def get_filter(self, idx):
        """Get filter by index"""
        if 0 <= idx < len(self.filters):
            return self.filters[idx]
        return None
        
    def get_filter_names(self):
        """Get list of filter names"""
        return [f'a={f.a:.3f}' for f in self.filters]
        
    def add_filter(self, a):
        """Add new filter with coefficient a"""
        if 0 <= a <= 1:
            self.filters.append(AllPassFilter(a))
            return True
        return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FilterDesignApp()
    window.show()
    sys.exit(app.exec_())
