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

        self.drag_btn = QPushButton("Drag Mode")
        self.drag_btn.setCheckable(True)  # Make button toggleable
        self.drag_btn.clicked.connect(lambda: self.set_mode('drag'))
        self.drag_btn.setStyleSheet(button_style)

        # Add to layout after other buttons
        left_layout.addWidget(self.drag_btn)
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
        self.z_plane_canvas.mpl_connect('button_press_event', self.on_click)
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
        # Initialize frequency response plots
        self.mag_ax = self.freq_figure.add_subplot(211)
        self.phase_ax = self.freq_figure.add_subplot(212)
        right_layout.addWidget(self.freq_canvas)
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        layout.addWidget(left_panel)
        layout.addWidget(center_panel)
        layout.addWidget(right_panel)
        
        self.initialize_plots()

        
        
        
        # Initialize signal processing variables
        from collections import deque
        self.max_samples = 10000
        self.input_signal = deque(maxlen=self.max_samples)
        self.output_signal = deque(maxlen=self.max_samples)
        self.phase_corrected_signal = deque(maxlen=self.max_samples)
        self.buffer_index = 0
        self.last_time = time.time()
        self.last_mouse_pos = None

        self.phase_corrected_signal = []
        self.processing_speed = 50
        self.last_mouse_y = None

        self.process_timer = QTimer()
        self.process_timer.timeout.connect(self.process_next_sample)
        self.process_timer.start(20)

        signal_panel = self.setup_signal_panel()
        all_pass_panel = self.setup_all_pass_panel()
        layout.addWidget(all_pass_panel)
        layout.addWidget(signal_panel)
        
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
        # Update plot styling for dark mode
        self.z_ax = self.z_plane_figure.add_subplot(111, facecolor=PLOT_BG)
        circle = Circle((0, 0), 1, fill=False, color=PLOT_TEXT)
        self.z_ax.add_artist(circle)
        self.z_ax.grid(True, color=PLOT_GRID)
        self.z_ax.set_aspect('equal')
        self.z_ax.set_xlim(-2, 2)
        self.z_ax.set_ylim(-2, 2)
        self.z_ax.set_xlabel('Real', color=PLOT_TEXT)
        self.z_ax.set_ylabel('Imaginary', color=PLOT_TEXT)
        self.z_ax.set_title('Z-Plane', color=PLOT_TEXT)
        self.z_ax.tick_params(colors=PLOT_TEXT)

        self.z_plane_canvas.mpl_connect('button_press_event', self.on_press)
        self.z_plane_canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.z_plane_canvas.mpl_connect('button_release_event', self.on_release)
            
        # Update frequency response plots styling
        for ax in [self.mag_ax, self.phase_ax]:
            ax.set_facecolor(PLOT_BG)
            ax.grid(True, color=PLOT_GRID)
            ax.tick_params(colors=PLOT_TEXT)
            
            for spine in ax.spines.values():
                spine.set_color(PLOT_TEXT)

    def set_mode(self, mode):
        """Sets the current tool mode (zero/pole/drag) and updates button states"""
        # Deactivate current mode if clicking active mode
        if self.current_mode == mode:
            self.current_mode = None
            self.add_zero_btn.setChecked(False)
            self.add_pole_btn.setChecked(False)
            self.drag_btn.setChecked(False)
        else:
            # Activate new mode
            self.current_mode = mode
            self.add_zero_btn.setChecked(mode == 'zero')
            self.add_pole_btn.setChecked(mode == 'pole')
            self.drag_btn.setChecked(mode == 'drag')
            
        # Update cursor based on mode
        if self.current_mode == 'drag':
            self.z_plane_canvas.setCursor(Qt.OpenHandCursor)
        elif self.current_mode:
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

    def on_click(self, event):
        if event.inaxes != self.z_ax:
            return
            
        x, y = event.xdata, event.ydata
        changed = False
        
        # Check if clicking near existing point to delete
        for i, zero in enumerate(self.zeros):
            if abs(zero.real - x) < 0.1 and abs(zero.imag - y) < 0.1:
                self.zeros.pop(i)
                changed = True
                break
                
        for i, pole in enumerate(self.poles):
            if abs(pole.real - x) < 0.1 and abs(pole.imag - y) < 0.1:
                self.poles.pop(i)
                changed = True
                break
        
        # Add new point if not deleting
        if not changed and self.current_mode:
            if self.current_mode == 'zero':
                self.zeros.append(complex(x, y))
                if self.conjugate_check.isChecked():
                    self.zeros.append(complex(x, -y))
                changed = True
            elif self.current_mode == 'pole':
                self.poles.append(complex(x, y))
                if self.conjugate_check.isChecked():
                    self.poles.append(complex(x, -y))
                changed = True
                
        if changed:
            self.add_to_history()
            self.update_plots()
        
    def update_plots(self):
        self.z_ax.clear()
        
        # Redraw unit circle and grid
        circle = Circle((0, 0), 1, fill=False, color='black')
        self.z_ax.add_artist(circle)
        self.z_ax.grid(True)
        self.z_ax.set_aspect('equal')
        self.z_ax.set_xlim(-2, 2)
        self.z_ax.set_ylim(-2, 2)
        
        # Plot zeros and poles
        for zero in self.zeros:
            self.z_ax.plot(zero.real, zero.imag, 'bo', markersize=10)
        for pole in self.poles:
            self.z_ax.plot(pole.real, pole.imag, 'rx', markersize=10)
            
        self.z_plane_canvas.draw()
        self.update_frequency_response()
        
    def update_frequency_response(self):
        """Update both magnitude and phase frequency responses"""
        # Calculate frequency points
        w = np.linspace(0, np.pi, 1000)
        z = np.exp(1j * w)
        
        # Calculate transfer function
        H = np.ones_like(z, dtype=complex)
        for zero in self.zeros:
            H *= (z - zero)
        for pole in self.poles:
            H /= (z - pole)
            
        # Calculate magnitude and phase
        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.unwrap(np.angle(H, deg=True))
        
        # Clear previous plots
        self.mag_ax.clear()
        self.phase_ax.clear()
        
        # Plot magnitude response
        self.mag_ax.plot(w/np.pi, mag_db, 'w-', linewidth=2)
        self.mag_ax.set_ylabel('Magnitude (dB)', color=PLOT_TEXT)
        self.mag_ax.grid(True, color=PLOT_GRID)
        self.mag_ax.set_title('Magnitude Response', color=PLOT_TEXT)
        
        # Plot phase response
        self.phase_ax.plot(w/np.pi, phase_deg, 'w-', linewidth=2)
        self.phase_ax.set_xlabel('Normalized Frequency (×π rad/sample)', color=PLOT_TEXT)
        self.phase_ax.set_ylabel('Phase (degrees)', color=PLOT_TEXT)
        self.phase_ax.grid(True, color=PLOT_GRID)
        self.phase_ax.set_title('Phase Response', color=PLOT_TEXT)
        
        # Update styling for both plots
        for ax in [self.mag_ax, self.phase_ax]:
            ax.set_facecolor(PLOT_BG)
            ax.grid(True, color=PLOT_GRID)
            ax.tick_params(colors=PLOT_TEXT)
            
            for spine in ax.spines.values():
                spine.set_color(PLOT_TEXT)
                
        # Adjust layout and draw
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
        """Save filter coefficients to .flt file"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Filter", 
            "", 
            "Filter Files (*.flt)"  # Remove All Files option to enforce .flt
        )
        
        if filename:
            # Add .flt extension if not present
            if not filename.endswith('.flt'):
                filename += '.flt'
                
            data = {
                'zeros': self.zeros,
                'poles': self.poles
            }
            
            # Use json to save as plain text instead of numpy binary
            with open(filename, 'w') as f:
                json.dump({
                    'zeros': [(z.real, z.imag) for z in self.zeros],
                    'poles': [(p.real, p.imag) for p in self.poles]
                }, f, indent=2)
            
    def load_filter(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Filter", "", "Filter Files (*.flt);;All Files (*)"
        )
        if filename:
            data = np.load(filename, allow_pickle=True).item()
            self.zeros = data['zeros']
            self.poles = data['poles']
            self.update_plots()
            
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

        
    def on_press(self, event):
        """Handle mouse press for dragging and adding points"""
        if event.inaxes != self.z_ax:
            return

        x, y = event.xdata, event.ydata
            
        # Left click
        if event.button == 1:
            # Check for dragging if in drag mode
            if self.current_mode == 'drag':
                # Check zeros then poles for dragging
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
                        
            # Add new point if in add mode
            elif self.current_mode in ['zero', 'pole']:
                if self.current_mode == 'zero':
                    self.zeros.append(complex(x, y))
                    if self.conjugate_check.isChecked():
                        self.zeros.append(complex(x, -y))
                else:  # pole mode
                    self.poles.append(complex(x, y))
                    if self.conjugate_check.isChecked():
                        self.poles.append(complex(x, -y))
                self.add_to_history()
                self.update_plots()

    def on_motion(self, event):
        """Handle dragging poles/zeros"""
        if not self.dragging or event.inaxes != self.z_ax:
            return
                
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
                
        # Update position of dragged point
        if self.drag_type == 'zero':
            self.zeros[self.drag_target] = complex(x, y)
            if self.conjugate_check.isChecked():
                conj_idx = self.find_conjugate(self.drag_target, self.zeros)
                if conj_idx is not None:
                    self.zeros[conj_idx] = complex(x, -y)
        else:  # pole
            self.poles[self.drag_target] = complex(x, y)
            if self.conjugate_check.isChecked():
                conj_idx = self.find_conjugate(self.drag_target, self.poles)
                if conj_idx is not None:
                    self.poles[conj_idx] = complex(x, -y)
                        
        self.update_plots()

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


    def find_conjugate(self, idx, points):
        """Find index of conjugate pair for given point"""
        point = points[idx]
        for i, p in enumerate(points):
            if i != idx and abs(p.real - point.real) < 0.01 and abs(p.imag + point.imag) < 0.01:
                return i
        return None
    

    def generate_c_code(self):
        """
        Generate and save C code for the designed filter to a file.
        """
        file_name = "filter_design.c"  

        c_code = f"""\
#include <stdio.h>
#include <math.h>

#define NUM_ZEROS {len(self.zeros)}
#define NUM_POLES {len(self.poles)}

double zeros[NUM_ZEROS] = {{{', '.join([f'{z.real:.6f} + {z.imag:.6f}i' for z in self.zeros])}}};
double poles[NUM_POLES] = {{{', '.join([f'{p.real:.6f} + {p.imag:.6f}i' for p in self.poles])}}};

void apply_filter(double *input, double *output, int length) {{
    for (int i = 0; i < length; i++) {{
        output[i] = input[i];
    }}
}}

int main() {{
    printf("Filter Design Loaded\\n");
    return 0;
}}
    """

        # Save the generated C code to a file.
        with open(file_name, "w") as file:
            file.write(c_code)
        
        print(f"C code saved to {file_name}.")


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
        """Process next input sample through filter chain"""
        if not self.input_signal:
            return
            
        try:
            # Get latest sample
            x = self.input_signal[-1]
            
            # Apply main filter
            y = self.apply_selected_filter(x)
            self.output_signal.append(float(y))
            
            # Apply all-pass filters if enabled
            if self.all_pass_enabled.isChecked():
                y = self.apply_all_pass_filters(y)
                
            self.phase_corrected_signal.append(float(y))
            
            # Update visualization periodically
            if len(self.input_signal) % 10 == 0:
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
        y = (self.drawing_area.height() - event.y()) / self.drawing_area.height() * 2 - 1
        
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
        """Apply the current filter to input sample x"""
        try:
            # Pass through if no filter defined
            if len(self.zeros) == 0 and len(self.poles) == 0:
                return x
                
            # Get coefficients based on selected form
            if self.direct_form.isChecked():
                coeffs = self.generate_direct_form_II()
                return self.apply_direct_form(x, coeffs)
            else:
                coeffs = self.generate_cascade_form()
                return self.apply_cascade_form(x, coeffs)
                
        except Exception as e:
            print(f"Error applying filter: {e}")
            return x

    def apply_direct_form(self, x, coeffs):
        try:
            b = np.array(coeffs['b'], dtype=float)
            a = np.array(coeffs['a'], dtype=float)
            
            # Current implementation may not be correctly handling state updates
            # Should be modified to:
            state_size = max(len(b), len(a)) - 1
            if self.direct_state is None or len(self.direct_state) != state_size:
                self.direct_state = np.zeros(state_size)
                
            # Direct Form II implementation
            w = x  # Input to state
            for i in range(1, len(a)):
                w = w - a[i] * self.direct_state[i-1]
                
            y = b[0] * w  # Output computation
            for i in range(1, len(b)):
                y = y + b[i] * self.direct_state[i-1]
                
            # Update state correctly
            self.direct_state = np.roll(self.direct_state, 1)
            self.direct_state[0] = w
            
            return float(y)

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
            
            # Ensure arrays
            b = np.array(b, dtype=float)
            a = np.array(a, dtype=float)
            
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
        if not self.input_signal or not hasattr(self, 'window_size_spin'):
            return
            
        # Get window size
        window_size = self.window_size_spin.value()
        
        # Convert deques to numpy arrays, ensuring same length
        input_data = np.array(list(self.input_signal))
        output_data = np.array(list(self.output_signal))
        phase_data = np.array(list(self.phase_corrected_signal))
        
        # Ensure all arrays have same length
        min_len = min(len(input_data), len(output_data), len(phase_data))
        if min_len == 0:
            return
            
        input_data = input_data[-min_len:]
        output_data = output_data[-min_len:]
        phase_data = phase_data[-min_len:]
        
        # Get data window
        if min_len > window_size:
            input_data = input_data[-window_size:]
            output_data = output_data[-window_size:]
            phase_data = phase_data[-window_size:]
        
        # Create time axis
        t = np.arange(len(input_data))
        
        # Update plots
        self.input_ax.clear()
        self.output_ax.clear()
        self.phase_ax.clear()
        
        # Plot synchronized data
        self.input_ax.plot(t, input_data, 'b-', label='Input')
        self.output_ax.plot(t, output_data, 'r-', label='Filtered') 
        self.phase_ax.plot(t, phase_data, 'g-', label='Phase Corrected')
        
        # Set proper labels and limits
        for ax in [self.input_ax, self.output_ax, self.phase_ax]:
            ax.grid(True)
            ax.legend()
            ax.set_ylim(min(input_data.min(), output_data.min(), phase_data.min()) - 0.1,
                        max(input_data.max(), output_data.max(), phase_data.max()) + 0.1)
        
        self.signal_figure.tight_layout()
        self.signal_canvas.draw()

    def export_filter(self):
        # Get current implementation type
        if self.direct_form.isChecked():
            coeffs = self.generate_direct_form_II()
        else:
            coeffs = self.generate_cascade_form()
            
        # Export to file
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Filter", "", "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump({
                    'type': 'direct' if self.direct_form.isChecked() else 'cascade',
                    'coefficients': coeffs
                }, f)


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

    def on_all_pass_enabled(self, state):
        """Handle enabling/disabling all-pass filters"""
        self.all_pass_list.setEnabled(state == Qt.Checked)
        self.a_input.setEnabled(state == Qt.Checked)
        if state == Qt.Checked:
            self.process_signal()  # Update signal with filters

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
            else:
                QMessageBox.warning(self, "Invalid Input", 
                                "Coefficient must be between 0 and 1")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", 
                            "Please enter a valid number")
    
    def setup_signal_panel(self):
        panel = QGroupBox("Real-time Processing")
        layout = QVBoxLayout()
        
        # Speed control with wider range
        speed_layout = QHBoxLayout()
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_value_label = QLabel("50 samples/second")
        self.speed_slider.valueChanged.connect(self.update_speed)
        
        speed_layout.addWidget(QLabel("Processing Speed:"))
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(self.speed_value_label)
        
        # Signal visualization controls
        viz_layout = QHBoxLayout()
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(100, 1000)
        self.window_size_spin.setValue(200)
        self.window_size_spin.valueChanged.connect(self.update_visualization)
        
        viz_layout.addWidget(QLabel("Window Size:"))
        viz_layout.addWidget(self.window_size_spin)
        
        # Mouse input area
        self.drawing_area = QWidget()
        self.drawing_area.setMinimumSize(300, 100)
        self.drawing_area.setStyleSheet(f"""
            QWidget {{
                background-color: {DARK_SECONDARY};
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        self.drawing_area.setMouseTracking(True)
        self.drawing_area.installEventFilter(self)
        
        # Signal plots with three traces
        self.signal_figure = Figure(figsize=(6, 6))
        self.signal_canvas = FigureCanvas(self.signal_figure)
        
        self.input_ax = self.signal_figure.add_subplot(311)
        self.output_ax = self.signal_figure.add_subplot(312)
        self.phase_ax = self.signal_figure.add_subplot(313)
        
        # Add all widgets
        layout.addLayout(speed_layout)
        layout.addLayout(viz_layout)
        layout.addWidget(self.drawing_area)
        layout.addWidget(self.signal_canvas)
        
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
        if obj is self.drawing_area:
            if event.type() == event.MouseMove:
                self.handle_mouse_draw(event)
                return True
        return super().eventFilter(obj, event)

    def handle_mouse_draw(self, event):
        """Handle mouse movement for signal generation"""
        if not hasattr(self, 'last_pos'):
            self.last_pos = event.pos()
            self.last_time = time.time()
            return
            
        # Calculate mouse velocity for frequency
        dt = time.time() - self.last_time
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()
        velocity = np.sqrt(dx*dx + dy*dy) / dt
        
        # Convert position to signal value (-1 to 1)
        y = 1.0 - (2.0 * event.pos().y() / self.drawing_area.height())
        
        # Add to input buffer
        self.input_signal.append(float(y))
        
        # Update state
        self.last_pos = event.pos()
        self.last_time = time.time()
        
        # Process sample
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
    def __init__(self, a):
        self.a = float(a)  # Ensure float
        self.zero = 1/self.a  # Reciprocal for all-pass
        self.pole = self.a
        self.state = 0.0  # Single state variable
        
        
    def process(self, x):
        """Process one sample through all-pass filter"""
        try:
            # Direct Form I implementation
            w = float(x) - self.pole * self.state
            y = self.zero * w + self.state
            self.state = w  # Update state
            return y
        except Exception as e:
            print(f"Error in filter: {e}")
            return x
        
    def get_phase_response(self, w):
        z = np.exp(1j * w)
        H = (z - self.zero)/(1 - self.pole*z)
        return np.angle(H)

class AllPassLibrary:
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