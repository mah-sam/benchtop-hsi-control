"""
hsi_control_app.py

A professional, unified PyQt6 application for controlling an HSI system,
integrating camera and stage functionalities into a single user interface.
...
"""
import sys
import os
import json
import time
from datetime import datetime
import numpy as np
import cv2
import traceback
import h5py

# ### NEW DEPENDENCY ###
# pyqtgraph is used for high-performance, interactive plotting in the new slice viewer.
# It can be installed with: pip install pyqtgraph
try:
    import pyqtgraph as pg
except ImportError:
    print("CRITICAL: pyqtgraph is not installed. Please run 'pip install pyqtgraph'.")
    pg = None


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QFormLayout,
    QSpinBox, QMessageBox, QDoubleSpinBox, QCheckBox, QFileDialog, QScrollArea, QLineEdit,
    QDialog, QDialogButtonBox, QProgressBar, QSlider, QTreeWidget, QTreeWidgetItem, QTabWidget,
    QVBoxLayout, QTableWidget, QTableWidgetItem, QHBoxLayout, QPushButton, QHeaderView, QRadioButton,
    QSizePolicy, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap, QAction, QCloseEvent, QPainter, QColor, QPen, QPainterPath, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSlot, QTimer, QThread, pyqtSignal, QRect, QPoint, QRectF

# Assuming controllers are in a 'hardware' sub-directory
from hardware.camera_controller import CameraController
from hardware.stage_controller import StageController
from calibration_wizard.calibration_wizard import CalibrationWizard
from dedicated_camera_app import CameraTestApp
from core import transformers
from core import file_io

def get_application_path():
    """
    Returns the base path for the application, whether it's running as a script
    or as a frozen cx_Freeze/PyInstaller executable.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the base path is the executable's directory
        return os.path.dirname(sys.executable)
    else:
        # If run as a normal script, the base path is the script's directory
        return os.path.dirname(os.path.abspath(__file__))

# --- Define Robust Paths (as before) ---
_CURRENT_DIR = get_application_path()
ASSETS_DIR = os.path.join(_CURRENT_DIR, 'assets')
CONFIG_FILE = os.path.join(ASSETS_DIR, 'stage_config.json')
MASTER_CALIBRATION_FILE = os.path.join(ASSETS_DIR, 'master_calibration.json')

from PyQt6.QtWidgets import QLabel, QRubberBand
from PyQt6.QtCore import pyqtSignal, QRect, QPoint, Qt
from PyQt6.QtGui import QMouseEvent

class ClickableSpectrogramLabel(QLabel):
    """A QLabel that emits the image's pixel coordinates (row, col) on click."""
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_width = 0
        self.original_height = 0

    def set_original_size(self, width: int, height: int):
        """Stores the size of the source image to calculate coordinates."""
        self.original_width = width
        self.original_height = height

    def mousePressEvent(self, event: QMouseEvent):
        pixmap = self.pixmap()
        if not pixmap or pixmap.isNull() or self.original_width == 0 or self.original_height == 0:
            return

        widget_w, widget_h = self.width(), self.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        
        offset_x = (widget_w - pixmap_w) / 2.0
        offset_y = (widget_h - pixmap_h) / 2.0
        
        pixmap_x = event.pos().x() - offset_x
        pixmap_y = event.pos().y() - offset_y

        if 0 <= pixmap_x < pixmap_w and 0 <= pixmap_y < pixmap_h:
            image_x = int(pixmap_x * self.original_width / pixmap_w)
            image_y = int(pixmap_y * self.original_height / pixmap_h)
            self.pixel_clicked.emit(image_y, image_x) # Emit row, col

            
class DataCubeSliceViewerDialog(QDialog):
    """
    An advanced dialog for interactively viewing and analyzing data cube slices,
    featuring a Photoshop-style "Magic Wand" for region-of-interest (ROI) analysis
    using adjustable HSV thresholds on the full RGB preview with an explicit 'Apply' step.
    All features, including detailed metrics and UI controls, are fully implemented and restored.
    """
    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.roi_sidecar_path = os.path.splitext(filepath)[0] + '.roi.json'
        
        # Data members
        self.data_cube, self.metadata, self.wavelengths = None, None, None
        self.rgb_preview, self.hsv_preview = None, None
        self.global_avg_spectrum, self.current_slice_float = None, None
        
        # ROI members
        self.roi_mask, self.roi_seed_point, self.roi_seed_hsv = None, None, None
        self.is_roi_selection_mode = False
        
        self.setWindowTitle(f"Advanced Slice Analyzer - {os.path.basename(filepath)}")
        self.setMinimumSize(1200, 800)

        if pg is None:
            QMessageBox.critical(self, "Dependency Error", "The 'pyqtgraph' library is required. Please run: pip install pyqtgraph")
            QTimer.singleShot(0, self.reject)
            return

        self._setup_ui()
        self._connect_signals()

        if not self._load_data():
            QTimer.singleShot(0, self.reject)
        else:
            self._load_roi()

    def _load_data(self) -> bool:
        try:
            # Note: load_h5 loads the entire cube into memory. This is necessary for this type of analysis.
            cube, metadata, _, rgb_preview_from_file = file_io.load_h5(self.filepath)
            
            if 'wavelength' not in metadata: raise ValueError("HDF5 metadata is missing 'wavelength' key.")
            
            self.data_cube, self.metadata = cube, metadata
            self.wavelengths = np.array(metadata['wavelength'])
            
            if rgb_preview_from_file is not None:
                self.rgb_preview = rgb_preview_from_file
                self.hsv_preview = cv2.cvtColor(rgb_preview_from_file, cv2.COLOR_BGR2HSV)
            else:
                self.roi_group.setEnabled(False)
                self.roi_group.setToolTip("ROI analysis requires an embedded RGB preview.")

            if self.data_cube.size > 0: self.global_avg_spectrum = np.mean(self.data_cube, axis=(1, 2))

            self.slice_slider.setRange(0, self.data_cube.shape[2] - 1)
            self.slice_spinbox.setRange(0, self.data_cube.shape[2] - 1)
            
            if self.global_avg_spectrum is not None:
                self.plot_widget.plot(self.wavelengths, self.global_avg_spectrum, pen=pg.mkPen(color=(150, 150, 150), width=2, style=Qt.PenStyle.DashLine), name="Global Avg")
                self.plot_group.setTitle("Global Average Spectrum (Initial)")

            self._update_slice_view()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or process HDF5 data cube:\n\n{e}\n{traceback.format_exc()}")
            return False

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        left_scroll_area = QScrollArea(); left_scroll_area.setWidgetResizable(True); left_scroll_area.setMaximumWidth(450)
        left_panel_container = QWidget(); left_layout = QVBoxLayout(left_panel_container); left_scroll_area.setWidget(left_panel_container)
        
        self.slice_controls_group = QGroupBox("Slice-by-Slice Controls")
        controls_layout = QFormLayout(self.slice_controls_group)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal); self.slice_spinbox = QSpinBox()
        slice_layout = QHBoxLayout(); slice_layout.addWidget(self.slice_slider); slice_layout.addWidget(self.slice_spinbox)
        self.gain_spinbox = QDoubleSpinBox(); self.gain_spinbox.setRange(0.1, 500.0); self.gain_spinbox.setValue(1.0); self.gain_spinbox.setSingleStep(0.5); self.gain_spinbox.setSuffix("x")
        self.colormaps = {'Grayscale': -1, 'Viridis': cv2.COLORMAP_VIRIDIS, 'Jet': cv2.COLORMAP_JET, 'Hot': cv2.COLORMAP_HOT}
        self.colormap_combo = QComboBox(); self.colormap_combo.addItems(self.colormaps.keys())
        self.auto_contrast_button = QPushButton("Auto-Adjust Contrast")
        controls_layout.addRow("Scan Slice:", slice_layout); controls_layout.addRow("Display Gain:", self.gain_spinbox)
        controls_layout.addRow("Colormap:", self.colormap_combo); controls_layout.addRow(self.auto_contrast_button)
        
        self.roi_group = QGroupBox("Region of Interest (ROI) Analysis"); self.roi_group.setCheckable(True); self.roi_group.setChecked(False)
        roi_layout = QVBoxLayout(self.roi_group)
        roi_controls_layout = QFormLayout()
        self.roi_h_slider = QSlider(Qt.Orientation.Horizontal); self.roi_h_slider.setRange(0, 90); self.roi_h_slider.setValue(10)
        self.roi_s_slider = QSlider(Qt.Orientation.Horizontal); self.roi_s_slider.setRange(0, 128); self.roi_s_slider.setValue(40)
        self.roi_v_slider = QSlider(Qt.Orientation.Horizontal); self.roi_v_slider.setRange(0, 128); self.roi_v_slider.setValue(40)
        self.roi_h_label = QLabel("10"); self.roi_s_label = QLabel("40"); self.roi_v_label = QLabel("40")
        def create_slider_row(slider, label): row = QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.addWidget(slider); row.addWidget(label); return row
        roi_controls_layout.addRow("Hue Tolerance:", create_slider_row(self.roi_h_slider, self.roi_h_label))
        roi_controls_layout.addRow("Saturation Tolerance:", create_slider_row(self.roi_s_slider, self.roi_s_label))
        roi_controls_layout.addRow("Value Tolerance:", create_slider_row(self.roi_v_slider, self.roi_v_label))
        self.roi_apply_button = QPushButton("Apply ROI & Calculate Spectrum"); self.roi_apply_button.setStyleSheet("font-weight: bold;")
        roi_buttons_widget = QWidget(); roi_buttons_layout = QHBoxLayout(roi_buttons_widget); roi_buttons_layout.setContentsMargins(0,0,0,0)
        self.roi_clear_button = QPushButton("Clear"); self.roi_save_button = QPushButton("Save")
        roi_buttons_layout.addWidget(self.roi_clear_button); roi_buttons_layout.addWidget(self.roi_save_button); roi_buttons_layout.addStretch()
        roi_layout.addWidget(QLabel("1. Click image to set seed point.\n2. Adjust HSV tolerance sliders.\n3. Click 'Apply' to calculate spectrum."))
        roi_layout.addLayout(roi_controls_layout); roi_layout.addWidget(self.roi_apply_button); roi_layout.addWidget(roi_buttons_widget)
        
        self.plot_group = QGroupBox("Spectral Profile"); plot_layout = QVBoxLayout(self.plot_group)
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.plot_widget = pg.PlotWidget(); self.plot_widget.setLabel('bottom', "Wavelength (nm)"); self.plot_widget.setLabel('left', "Intensity (A.U.)"); self.plot_widget.showGrid(x=True, y=True); self.plot_widget.addLegend()
        self.peak_wavelength_label = QLabel("N/A"); self.peak_intensity_label = QLabel("N/A")
        self.snr_label = QLabel("N/A"); self.fwhm_label = QLabel("N/A"); self.centroid_label = QLabel("N/A")
        metrics_layout = QFormLayout()
        metrics_layout.addRow("Peak Wavelength:", self.peak_wavelength_label); metrics_layout.addRow("Peak Intensity:", self.peak_intensity_label)
        metrics_layout.addRow("SNR:", self.snr_label); metrics_layout.addRow("FWHM:", self.fwhm_label); metrics_layout.addRow("Centroid:", self.centroid_label)
        plot_layout.addWidget(self.plot_widget); plot_layout.addLayout(metrics_layout)

        left_layout.addWidget(self.slice_controls_group); left_layout.addWidget(self.roi_group); left_layout.addWidget(self.plot_group); left_layout.addStretch()
        
        self.image_label = ClickableSpectrogramLabel(); self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.image_label.setStyleSheet("background-color: #222; color: #888;"); self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        main_layout.addWidget(left_scroll_area); main_layout.addWidget(self.image_label, 1)

    def _connect_signals(self):
        self.slice_slider.valueChanged.connect(self.slice_spinbox.setValue); self.slice_spinbox.valueChanged.connect(self.slice_slider.setValue)
        self.slice_spinbox.valueChanged.connect(self._update_slice_view)
        self.gain_spinbox.valueChanged.connect(self._update_slice_view)
        self.colormap_combo.currentIndexChanged.connect(self._update_slice_view)
        self.auto_contrast_button.clicked.connect(self._auto_adjust_contrast)
        self.image_label.pixel_clicked.connect(self._handle_image_click)
        self.roi_group.toggled.connect(self._on_roi_toggled)
        self.roi_h_slider.valueChanged.connect(self._on_roi_hsv_changed); self.roi_s_slider.valueChanged.connect(self._on_roi_hsv_changed); self.roi_v_slider.valueChanged.connect(self._on_roi_hsv_changed)
        self.roi_clear_button.clicked.connect(self._clear_roi); self.roi_save_button.clicked.connect(self._save_roi); self.roi_apply_button.clicked.connect(self._on_apply_roi)

    def _on_roi_toggled(self, checked: bool):
        self.is_roi_selection_mode = checked
        self.slice_controls_group.setEnabled(not checked)
        if not checked:
            self._clear_roi()
        self.image_label.setCursor(Qt.CursorShape.CrossCursor if checked else Qt.CursorShape.ArrowCursor)
        self._update_slice_view()

    def _handle_image_click(self, row: int, col: int):
        if self.is_roi_selection_mode:
            if self.hsv_preview is None: return
            self.roi_seed_point = (col, row) # Store as (x, y) or (col, row)
            self.roi_seed_hsv = self.hsv_preview[row, col]
            self._update_roi_segmentation()
        else:
            self._update_spectrum_plot(col)

    def _on_roi_hsv_changed(self):
        self.roi_h_label.setText(str(self.roi_h_slider.value()))
        self.roi_s_label.setText(str(self.roi_s_slider.value()))
        self.roi_v_label.setText(str(self.roi_v_slider.value()))
        if self.roi_seed_point:
            self._update_roi_segmentation()

    def _update_roi_segmentation(self):
        if self.hsv_preview is None or self.roi_seed_point is None or self.roi_seed_hsv is None: return
        h_tol, s_tol, v_tol = self.roi_h_slider.value(), self.roi_s_slider.value(), self.roi_v_slider.value()
        seed_h, seed_s, seed_v = self.roi_seed_hsv
        h_min, h_max = int(seed_h) - h_tol, int(seed_h) + h_tol
        s_min, s_max = max(0, int(seed_s) - s_tol), min(255, int(seed_s) + s_tol)
        v_min, v_max = max(0, int(seed_v) - v_tol), min(255, int(seed_v) + v_tol)
        
        # Handle hue wrapping for red colors (around 0/179 in OpenCV)
        if h_min < 0:
            lower1 = np.array([0, s_min, v_min]); upper1 = np.array([h_max, s_max, v_max])
            lower2 = np.array([180 + h_min, s_min, v_min]); upper2 = np.array([179, s_max, v_max])
            color_mask = cv2.inRange(self.hsv_preview, lower1, upper1) | cv2.inRange(self.hsv_preview, lower2, upper2)
        elif h_max > 179:
            lower1 = np.array([h_min, s_min, v_min]); upper1 = np.array([179, s_max, v_max])
            lower2 = np.array([0, s_min, v_min]); upper2 = np.array([h_max - 180, s_max, v_max])
            color_mask = cv2.inRange(self.hsv_preview, lower1, upper1) | cv2.inRange(self.hsv_preview, lower2, upper2)
        else:
            lower = np.array([h_min, s_min, v_min]); upper = np.array([h_max, s_max, v_max])
            color_mask = cv2.inRange(self.hsv_preview, lower, upper)
            
        flood_mask = np.zeros((self.hsv_preview.shape[0] + 2, self.hsv_preview.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(color_mask.copy(), flood_mask, self.roi_seed_point, 255)
        self.roi_mask = flood_mask[1:-1, 1:-1].astype(bool)
        self._update_slice_view()

    @pyqtSlot()
    def _on_apply_roi(self):
        self._calculate_and_display_roi_spectrum()

    def _calculate_and_display_roi_spectrum(self):
        if self.roi_mask is None or not np.any(self.roi_mask) or self.data_cube is None:
            QMessageBox.warning(self, "No ROI", "Please define a valid ROI before applying.")
            return

        # 1. Get the dimensions of both coordinate systems
        # "Preview Space" (from the mask, which matches the RGB preview)
        preview_h, preview_w = self.roi_mask.shape

        # "Cube Space" (from the data cube)
        cube_spectral_h, cube_spatial_w, cube_slice_h = self.data_cube.shape

        # 2. Check for dimension mismatch that indicates a problem
        if preview_h != cube_slice_h:
            # This would be a very strange bug, but it's good practice to check.
            # The height (number of slices) should always match.
            QMessageBox.critical(self, "Dimension Error", 
                f"Mismatch in slice dimension between preview ({preview_h}) and data cube ({cube_slice_h}). Cannot calculate ROI.")
            return

        # 3. Get the coordinates of the ROI pixels in "Preview Space"
        # preview_y_coords corresponds to the slice axis
        # preview_x_coords corresponds to the spatial axis
        preview_y_coords, preview_x_coords = np.where(self.roi_mask)

        # 4. === CORE OF THE FIX: TRANSFORM COORDINATES ===
        # Calculate the scaling factor between the two coordinate systems.
        # This is the ratio of the cube's width to the preview's width.
        x_scale_factor = cube_spatial_w / preview_w

        # Apply the transformation. Convert preview X coordinates to cube X coordinates.
        # The Y coordinates (slices) don't need scaling as their dimension matches.
        cube_spatial_indices = (preview_x_coords * x_scale_factor).astype(np.int64)
        cube_slice_indices = preview_y_coords # No change needed

        # 5. Perform advanced indexing using the CORRECTED coordinates
        selected_spectra = self.data_cube[:, cube_spatial_indices, cube_slice_indices]

        # 6. Proceed with averaging and plotting as before
        if selected_spectra.size > 0:
            avg_spectrum = np.mean(selected_spectra, axis=1)
            self.plot_widget.clear()
            self.plot_widget.plot(self.wavelengths, avg_spectrum, pen=pg.mkPen(color=(50, 150, 50), width=2), name="ROI Avg")
            self.plot_group.setTitle(f"ROI Average Spectrum ({selected_spectra.shape[1]} pixels)")
            self._calculate_and_display_metrics(avg_spectrum)
        else:
            self._clear_roi()

    def _clear_roi(self):
        self.roi_mask, self.roi_seed_point, self.roi_seed_hsv = None, None, None
        self.plot_widget.clear()
        self.plot_group.setTitle("Spectral Profile")
        if self.global_avg_spectrum is not None:
            self.plot_widget.plot(self.wavelengths, self.global_avg_spectrum, pen=pg.mkPen(color=(150, 150, 150), width=2, style=Qt.PenStyle.DashLine), name="Global Avg")
        self._calculate_and_display_metrics(None)
        self._update_slice_view()

    def _save_roi(self):
        if self.roi_seed_point is None:
            QMessageBox.warning(self, "No ROI", "Please select an ROI on the image before saving.")
            return
        roi_data = {
            "seed_point_col_row": self.roi_seed_point,
            "h_tolerance": self.roi_h_slider.value(),
            "s_tolerance": self.roi_s_slider.value(),
            "v_tolerance": self.roi_v_slider.value()
        }
        try:
            with open(self.roi_sidecar_path, 'w') as f:
                json.dump(roi_data, f, indent=4)
            QMessageBox.information(self, "Success", f"ROI settings saved to:\n{os.path.basename(self.roi_sidecar_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save ROI settings.\n\n{e}")

    def _load_roi(self):
        if not os.path.exists(self.roi_sidecar_path) or self.hsv_preview is None: return
        try:
            with open(self.roi_sidecar_path, 'r') as f: roi_data = json.load(f)
            self.roi_group.setChecked(True)
            self.roi_seed_point = tuple(roi_data["seed_point_col_row"])
            self.roi_seed_hsv = self.hsv_preview[self.roi_seed_point[1], self.roi_seed_point[0]]
            self.roi_h_slider.setValue(roi_data.get("h_tolerance", 10))
            self.roi_s_slider.setValue(roi_data.get("s_tolerance", 40))
            self.roi_v_slider.setValue(roi_data.get("v_tolerance", 40))
            self._update_roi_segmentation()
            self._on_apply_roi()
        except Exception as e:
            QMessageBox.warning(self, "ROI Load Error", f"Could not load or apply saved ROI settings.\n\n{e}")

    def _update_slice_view(self):
        # ROI mode displays the RGB preview with a mask overlay
        if self.is_roi_selection_mode:
            if self.rgb_preview is None: return
            display_img = self.rgb_preview.copy()
            if self.roi_mask is not None:
                # Create a red overlay and apply it only where the mask is NOT active
                red_overlay = np.full(display_img.shape, (0, 0, 255), dtype=np.uint8)
                display_img[~self.roi_mask] = cv2.addWeighted(display_img[~self.roi_mask], 0.6, red_overlay[~self.roi_mask], 0.4, 0)
            
            h, w, ch = display_img.shape
            q_image = QImage(display_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.image_label.set_original_size(w, h)
        # Normal mode displays a single slice of the data cube
        else:
            if self.data_cube is None: return
            slice_idx = self.slice_spinbox.value()
            self.current_slice_float = self.data_cube[:, :, slice_idx].astype(np.float32)
            processed_slice = self.current_slice_float * self.gain_spinbox.value()
            cv2.normalize(processed_slice, processed_slice, 0, 255, cv2.NORM_MINMAX)
            slice_8bit = processed_slice.astype(np.uint8)
            
            colormap_code = self.colormaps[self.colormap_combo.currentText()]
            if colormap_code == -1: # Grayscale
                display_img = cv2.cvtColor(slice_8bit, cv2.COLOR_GRAY2BGR)
            else:
                display_img = cv2.applyColorMap(slice_8bit, colormap_code)
            
            h, w, ch = display_img.shape
            q_image = QImage(display_img.data, w, h, ch * w, QImage.Format.Format_BGR888)
            self.image_label.set_original_size(w, h)
            
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))

    def _update_spectrum_plot(self, col_idx: int):
        if self.data_cube is None: return
        self.plot_group.setTitle("Single-Pixel Spectrum")
        slice_idx = self.slice_spinbox.value()
        spectral_strip = self.data_cube[:, col_idx, slice_idx]
        self.plot_widget.clear()
        self.plot_widget.plot(self.wavelengths, spectral_strip, pen=pg.mkPen(color=(0, 100, 200), width=2), name="Single Pixel")
        if self.global_avg_spectrum is not None:
            self.plot_widget.plot(self.wavelengths, self.global_avg_spectrum, pen=pg.mkPen(color=(150, 150, 150), width=2, style=Qt.PenStyle.DashLine), name="Global Avg")
        self._calculate_and_display_metrics(spectral_strip)

    def _calculate_and_display_metrics(self, spectrum: np.ndarray | None):
        if spectrum is None or spectrum.size == 0 or self.wavelengths is None:
            for label in [self.peak_wavelength_label, self.peak_intensity_label, self.snr_label, self.fwhm_label, self.centroid_label]: label.setText("N/A")
            return
            
        peak_idx = np.argmax(spectrum)
        peak_val, peak_wav = spectrum[peak_idx], self.wavelengths[peak_idx]
        self.peak_wavelength_label.setText(f"{peak_wav:.2f} nm")
        self.peak_intensity_label.setText(f"{peak_val:.2f}")
        
        mean_val, std_val = np.mean(spectrum), np.std(spectrum)
        self.snr_label.setText(f"{mean_val / std_val:.2f} (Mean/StdDev)" if std_val > 1e-6 else "N/A")
        
        total_intensity = np.sum(spectrum)
        self.centroid_label.setText(f"{np.sum(spectrum * self.wavelengths) / total_intensity:.2f} nm" if total_intensity > 1e-6 else "N/A")
        
        try:
            min_val = np.min(spectrum)
            half_max = min_val + (peak_val - min_val) / 2.0
            above_half_max = np.where(spectrum > half_max)[0]
            if len(above_half_max) < 2: raise ValueError("Not enough points for FWHM.")
            
            left_idx, right_idx = above_half_max[0], above_half_max[-1]
            if left_idx == 0 or right_idx >= len(spectrum) - 1: raise ValueError("Peak at boundary.")
            
            # Linear interpolation for FWHM
            y1, y2 = spectrum[left_idx-1], spectrum[left_idx]; w1, w2 = self.wavelengths[left_idx-1], self.wavelengths[left_idx]
            left_wav = w1 + (w2 - w1) * (half_max - y1) / (y2 - y1)
            y1, y2 = spectrum[right_idx], spectrum[right_idx+1]; w1, w2 = self.wavelengths[right_idx], self.wavelengths[right_idx+1]
            right_wav = w1 + (w2 - w1) * (half_max - y1) / (y2 - y1)
            
            self.fwhm_label.setText(f"{right_wav - left_wav:.2f} nm")
        except (ValueError, IndexError):
            self.fwhm_label.setText("N/A")

    def _auto_adjust_contrast(self):
        if self.is_roi_selection_mode or self.current_slice_float is None: return
        min_val, max_val = np.percentile(self.current_slice_float, [2, 98])
        if max_val > min_val:
            # Adjust gain to map the 2-98 percentile range to the full 0-255 display range
            self.gain_spinbox.setValue(255.0 / (max_val - min_val))


class ManageLabelsDialog(QDialog):
    """A dialog to let users add, edit, and remove custom label fields."""
    # Emits the new list of label fields when saved.
    # Format: [{"key": "Sample Name", "description": "A unique identifier..."}, ...]
    label_fields_updated = pyqtSignal(list)

    def __init__(self, current_fields: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Custom Label Fields")
        self.setMinimumSize(500, 300)

        self.fields = current_fields

        # --- UI Setup ---
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(self.fields), 2)
        self.table.setHorizontalHeaderLabels(["Label Key", "Description (for placeholder text)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        for row, field in enumerate(self.fields):
            self.table.setItem(row, 0, QTableWidgetItem(field.get("key", "")))
            self.table.setItem(row, 1, QTableWidgetItem(field.get("description", "")))

        button_layout = QHBoxLayout()
        add_button = QPushButton("Add New Field")
        delete_button = QPushButton("Delete Selected Field")
        save_button = QPushButton("Save and Close")
        
        button_layout.addWidget(add_button)
        button_layout.addWidget(delete_button)
        button_layout.addStretch()
        button_layout.addWidget(save_button)
        
        layout.addWidget(self.table)
        layout.addLayout(button_layout)

        # --- Connections ---
        add_button.clicked.connect(self._add_row)
        delete_button.clicked.connect(self._delete_row)
        save_button.clicked.connect(self._save_and_close)

    def _add_row(self):
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        self.table.setItem(row_count, 0, QTableWidgetItem("New Key"))
        self.table.setItem(row_count, 1, QTableWidgetItem("New description..."))

    def _delete_row(self):
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def _save_and_close(self):
        new_fields = []
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            desc_item = self.table.item(row, 1)
            if key_item and key_item.text(): # Only save if key is not empty
                new_fields.append({
                    "key": key_item.text(),
                    "description": desc_item.text() if desc_item else ""
                })
        self.label_fields_updated.emit(new_fields)
        self.accept()

class DataCubeViewerDialog(QDialog):
    """
    A dialog for viewing the contents of a saved HDF5 data cube file.
    It dynamically displays metadata and labels, making it robust to future changes.
    """
    def __init__(self, filepath: str, parent=None):
        super().__init__(parent)
        self.filepath = filepath
        self.setWindowTitle(f"Data Cube Viewer - {os.path.basename(filepath)}")
        self.setMinimumSize(1000, 700)

        self._setup_ui()
        self._load_and_display_data()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left side: Image Preview
        image_group = QGroupBox("Embedded RGB Preview")
        image_layout = QVBoxLayout(image_group)
        self.image_label = QLabel("No RGB preview found in file.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #222; color: #888;")
        image_layout.addWidget(self.image_label)
        
        # Right side: Metadata and Labels in Tabs
        self.tabs = QTabWidget()
        self.metadata_tree = QTreeWidget()
        self.metadata_tree.setHeaderLabels(["Property", "Value"])
        self.metadata_tree.setColumnWidth(0, 200)
        
        self.labels_tree = QTreeWidget()
        self.labels_tree.setHeaderLabels(["Property", "Value"])
        self.labels_tree.setColumnWidth(0, 200)

        self.tabs.addTab(self.metadata_tree, "Metadata")
        self.tabs.addTab(self.labels_tree, "Labels")

        main_layout.addWidget(image_group, 2) # Give image more stretch
        main_layout.addWidget(self.tabs, 1)

    def _load_and_display_data(self):
        """Loads data from the H5 file and populates the UI widgets."""
        try:
            # ### FIX: Call the new, efficient function instead of the slow one ###
            metadata, labels, rgb_preview = file_io.load_h5_preview_and_metadata(self.filepath)

            # Display RGB Preview
            if rgb_preview is not None:
                h, w, ch = rgb_preview.shape
                q_image = QImage(rgb_preview.data, w, h, ch * w, QImage.Format.Format_BGR888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), 
                                                         Qt.AspectRatioMode.KeepAspectRatio, 
                                                         Qt.TransformationMode.SmoothTransformation))
            
            # Populate the metadata and labels trees
            self._populate_tree(self.metadata_tree, metadata)
            self._populate_tree(self.labels_tree, labels if labels else {"Info": "No labels present in this file."})

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or parse HDF5 file:\n\n{e}")
            # Close the dialog if the file is invalid
            QTimer.singleShot(0, self.reject)

    def _populate_tree(self, tree_widget: QTreeWidget, data: dict):
        """Recursively populates a QTreeWidget from a nested dictionary."""
        tree_widget.clear()
        self._add_tree_items(tree_widget.invisibleRootItem(), data)
        tree_widget.expandAll()

    def _add_tree_items(self, parent_item: QTreeWidgetItem, data: any):
        """Recursive helper to add dictionary items to the tree."""
        if isinstance(data, dict):
            for key, value in data.items():
                child_item = QTreeWidgetItem([str(key)])
                parent_item.addChild(child_item)
                self._add_tree_items(child_item, value)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                # Display list items with their index
                child_item = QTreeWidgetItem([f"[{i}]"])
                parent_item.addChild(child_item)
                self._add_tree_items(child_item, value)
        else:
            # This is a leaf node
            parent_item.setText(1, str(data))

class InteractiveCropLabel(QLabel):
    """
    A custom QLabel that allows a user to draw a rectangular selection box (rubber band)
    with the mouse. It emits a signal with the final selection rectangle.
    """
    crop_rect_changed = pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self.current_crop_rect = QRect()
        self._is_drawing_enabled = True

    def set_drawing_enabled(self, enabled: bool):
        """Enable or disable the ability to draw a new crop box."""
        self._is_drawing_enabled = enabled
        if not enabled and self.rubber_band.isVisible():
            self.rubber_band.hide()
            self.origin = QPoint()

    def mousePressEvent(self, event: QMouseEvent):
        """Starts the rubber band selection process."""
        if event.button() == Qt.MouseButton.LeftButton and self._is_drawing_enabled:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QPoint()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Updates the rubber band geometry as the mouse moves."""
        if not self.origin.isNull() and self._is_drawing_enabled:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Finalizes the selection and emits the rectangle."""
        if event.button() == Qt.MouseButton.LeftButton and not self.origin.isNull() and self._is_drawing_enabled:
            self.rubber_band.hide()
            self.current_crop_rect = self.rubber_band.geometry()
            self.crop_rect_changed.emit(self.current_crop_rect)
            self.origin = QPoint() # Reset for next time

    def clear_selection(self):
        """Clears the current crop rectangle."""
        self.current_crop_rect = QRect()
        self.crop_rect_changed.emit(self.current_crop_rect)

class HDF5SaveThread(QThread):
    """
    Saves HDF5 files with user-selectable compression, filters, and batching.
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress_update = pyqtSignal(int, int, str)

    def __init__(self, data_package: dict, parent=None):
        super().__init__(parent)
        self.data_package = data_package

    def run(self):
        filepath = self.data_package.get("filepath")
        if not filepath:
            self.error.emit("Filepath missing from data package.")
            return

        try:
            data_cube = self.data_package.get("data_cube")
            rgb_preview = self.data_package.get("rgb_preview")
            compression_algo = self.data_package.get("compression_algo", "gzip")
            compression_level = self.data_package.get("compression_level", 4)
            shuffle = self.data_package.get("shuffle_enabled", False)
            fletcher32 = self.data_package.get("fletcher32_enabled", False)
            batching_enabled = self.data_package.get("batching_enabled", True)
            batch_size = self.data_package.get("batch_size", 32)

            with h5py.File(filepath, 'w') as f:
                f.attrs['metadata'] = json.dumps(self.data_package.get("metadata"))
                labels = self.data_package.get("labels")
                if labels:
                    f.attrs['labels'] = json.dumps(labels)

                if compression_algo == 'None' or compression_algo is None:
                    # --- PATH 1: UNCOMPRESSED (FAST, NO PROGRESS) ---
                    self.progress_update.emit(0, 1, "Writing uncompressed data cube...")
                    f.create_dataset('cube', data=data_cube)
                    if rgb_preview is not None:
                        f.create_dataset('rgb_preview', data=rgb_preview)
                    self.progress_update.emit(1, 1, "Finished writing.")
                    final_algo = 'None'
                else:
                    # --- PATH 2: COMPRESSED (BATCHED OR SLICE-BY-SLICE) ---
                    final_algo = compression_algo
                    dset_kwargs = {
                        'chunks': True,
                        'compression': final_algo,
                        'shuffle': shuffle,
                        'fletcher32': fletcher32
                    }
                    if final_algo == 'gzip':
                        dset_kwargs['compression_opts'] = compression_level

                    num_slices = data_cube.shape[2]
                    self.progress_update.emit(0, num_slices, f"Creating compressed data cube dataset...")
                    
                    dset = f.create_dataset('cube', shape=data_cube.shape, dtype=data_cube.dtype, **dset_kwargs)

                    if batching_enabled:
                        for i in range(0, num_slices, batch_size):
                            start_idx = i
                            end_idx = min(i + batch_size, num_slices)
                            dset[:, :, start_idx:end_idx] = data_cube[:, :, start_idx:end_idx]
                            self.progress_update.emit(end_idx, num_slices, f"Compressing batch... ({end_idx}/{num_slices})")
                    else: # Slice-by-slice
                        for i in range(num_slices):
                            dset[:, :, i] = data_cube[:, :, i]
                            self.progress_update.emit(i + 1, num_slices, f"Compressing slice... ({i+1}/{num_slices})")

                    if rgb_preview is not None:
                        self.progress_update.emit(0, 1, f"Writing compressed RGB preview...")
                        f.create_dataset('rgb_preview', data=rgb_preview, **dset_kwargs)
                        self.progress_update.emit(1, 1, f"RGB preview finished.")

            self.finished.emit(f"Successfully saved to {filepath} using '{final_algo}' compression.")
        except Exception as e:
            error_str = f"Failed to save HDF5 file: {e}\n{traceback.format_exc()}"
            self.error.emit(error_str)

            
class DataCubeAcquisitionThread(QThread):
    """
    PRODUCER THREAD: Acquires frames, assembles the data cube and RGB preview
    in memory based on user settings, and then hands off the data for saving.
    """
    progress_update = pyqtSignal(int, int, str)
    rgb_preview_update = pyqtSignal(object)
    data_ready_for_saving = pyqtSignal(dict)
    # ### MODIFIED: Added a 'dict' to the signal to carry the labels ###
    scan_finished = pyqtSignal(str, object, dict, dict) # Path, RGB Preview, Metadata, Labels
    error_occurred = pyqtSignal(str)

    def __init__(self, camera_controller, stage_controller, config: dict, parent=None):
        super().__init__(parent)
        self.camera = camera_controller
        self.stage = stage_controller
        self.config = config
        self._is_interrupted = False

    @staticmethod
    def get_pixel_for_wavelength(wavelength: float, coeffs: list, max_pixel: int) -> int | None:
        poly_func = np.poly1d(coeffs)
        roots = (poly_func - wavelength).roots
        for root in roots:
            if np.isreal(root):
                real_root = int(round(np.real(root)))
                if 0 <= real_root < max_pixel:
                    return real_root
        return None

    def run(self):
        try:
            # ... (code for scanning and data cube assembly is unchanged) ...
            self.progress_update.emit(0, 1, "Preparing for scan...")
            calib_data = self.config['calibration_data']
            spectral_calib = calib_data.get("calibration_steps", {}).get('spectral')
            if not spectral_calib or 'coefficients' not in spectral_calib:
                raise ValueError("Spectral calibration is missing from the master file.")
            
            start_pos, end_pos = self.config['start_position'], self.config['end_position']
            speed_steps_per_sec = self.config['scan_speed']
            scan_duration_s = self.config['scan_duration_seconds']
            scan_fps = self.config['scan_fps']
            num_frames_to_capture = int(scan_duration_s * scan_fps)
            if num_frames_to_capture < 1:
                raise ValueError("Calculated frames to capture is zero. Check scan parameters.")
            
            self.progress_update.emit(0, num_frames_to_capture, f"Scan will capture {num_frames_to_capture} frames...")
            self.camera.cam.BeginAcquisition()
            self.stage.move_to(end_pos, speed_steps_per_sec)
            
            collected_frames_for_cube = []
            collected_rgb_strips = []
            frame_interval_s = 1.0 / scan_fps
            start_time = time.monotonic()
            
            last_frame_processed = None
            embed_rgb_preview = self.config.get('embed_rgb_preview', True)

            for i in range(num_frames_to_capture):
                if self._is_interrupted:
                    self.progress_update.emit(i, num_frames_to_capture, "Scan cancelled by user.")
                    break
                
                img_result = self.camera.cam.GetNextImage(2000)
                if img_result.IsIncomplete(): continue
                raw_frame = img_result.GetNDArray().copy()
                img_result.Release()
                
                corrected_frame = HsiControlApp.apply_geometric_corrections(raw_frame, self.config['calibration_data'])
                
                size_percent = self.config.get('output_size_percentage', 100)
                if size_percent < 100:
                    output_dims = calib_data['calibration_steps']['output_dimensions']
                    target_w = int(output_dims['width'] * size_percent / 100)
                    target_h = int(output_dims['height'] * size_percent / 100)
                    final_frame = cv2.resize(corrected_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    final_frame = corrected_frame
                
                collected_frames_for_cube.append(final_frame)
                last_frame_processed = final_frame

                if embed_rgb_preview:
                    rgb_sampling_params = {
                        'r_wavelength': self.config['r_wavelength'],
                        'g_wavelength': self.config['g_wavelength'],
                        'b_wavelength': self.config['b_wavelength'],
                        'thickness': self.config.get('sampling_thickness', 5),
                        'spectral_calib': spectral_calib,
                        'calib_output_height': calib_data['calibration_steps']['output_dimensions']['height']
                    }
                    single_row_image, _ = HsiControlApp._create_rgb_strip_from_frame(final_frame, rgb_sampling_params)
                    collected_rgb_strips.append(single_row_image)
                    
                    if collected_rgb_strips:
                        temp_rgb_image = np.vstack(collected_rgb_strips)
                        self.rgb_preview_update.emit(temp_rgb_image)
                
                self.progress_update.emit(i + 1, num_frames_to_capture, f"Captured frame {i+1}/{num_frames_to_capture}")
                sleep_duration_s = (start_time + (i + 1) * frame_interval_s) - time.monotonic()
                if sleep_duration_s > 0:
                    self.msleep(int(sleep_duration_s * 1000))
            
            if not collected_frames_for_cube: raise ValueError("No data was collected during the scan.")

            self.progress_update.emit(num_frames_to_capture, num_frames_to_capture, "Assembling data cube...")
            data_cube = np.stack(collected_frames_for_cube, axis=0).transpose(1, 2, 0)
            del collected_frames_for_cube
            
            final_rgb_preview = None
            if embed_rgb_preview and collected_rgb_strips:
                final_rgb_image_raw = np.vstack(collected_rgb_strips)
                del collected_rgb_strips

                if self.config.get('maintain_aspect_ratio', True):
                    scan_dist_mm = abs(end_pos - start_pos) * self.config['mm_per_step']
                    fov_width_mm = self.config['camera_fov_width_mm']
                    if scan_dist_mm > 0 and fov_width_mm > 0:
                        aspect_ratio = fov_width_mm / scan_dist_mm
                        h, _ = final_rgb_image_raw.shape[:2]
                        target_w = int(h * aspect_ratio)
                        MAX_IMAGE_WIDTH = 16384
                        if target_w > MAX_IMAGE_WIDTH:
                            self.progress_update.emit(num_frames_to_capture, num_frames_to_capture, f"Warning: Calculated width ({target_w}px) is extreme. Capping at {MAX_IMAGE_WIDTH}px.")
                            target_w = MAX_IMAGE_WIDTH
                        final_rgb_image_resized = cv2.resize(final_rgb_image_raw, (target_w, h), interpolation=cv2.INTER_AREA)
                    else:
                        final_rgb_image_resized = final_rgb_image_raw
                else:
                    final_rgb_image_resized = final_rgb_image_raw
                
                final_rgb_preview = cv2.normalize(final_rgb_image_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            self.progress_update.emit(num_frames_to_capture, num_frames_to_capture, "Generating metadata...")
            spectral_height, spatial_width, num_bands = data_cube.shape
            calib_output_h = calib_data['calibration_steps']['output_dimensions']['height']
            
            resized_rows = np.arange(spectral_height)
            if spectral_height > 1 and calib_output_h > 1:
                scale_factor = (calib_output_h - 1) / (spectral_height - 1)
                projected_rows_in_calib_space = resized_rows * scale_factor
            else:
                projected_rows_in_calib_space = resized_rows
            wavelengths = transformers.map_pixel_to_wavelength(projected_rows_in_calib_space, spectral_calib).tolist()

            rgb_sampling_info = {}
            if embed_rgb_preview:
                _, rgb_sampling_info = HsiControlApp._create_rgb_strip_from_frame(last_frame_processed, rgb_sampling_params)
            
            labels = self.config.get("labels", {})

            metadata = {
                'description': 'Hyperspectral data cube acquired from HSI Control App',
                'acquisition_date': datetime.now().isoformat(),
                'shape': (spectral_height, spatial_width, num_bands),
                'interleave': 'bsq', 'wavelength_units': 'nm',
                'wavelength': wavelengths,
                'rgb_preview_sampling': rgb_sampling_info,
                'scan_parameters': {k: v for k, v in self.config.items() if k not in ['calibration_data', 'save_path', 'labels']},
                'calibration_data_summary': { 'wizard_version': calib_data.get('wizard_version'), 'creation_date': calib_data.get('creation_date') }
            }
            
            data_package = {
                "filepath": self.config['save_path'], "data_cube": data_cube,
                "metadata": metadata, "labels": labels, "rgb_preview": final_rgb_preview,
                "compression_algo": self.config.get('compression_algo', 'gzip'),
                "compression_level": self.config.get('compression_level', 4),
                "shuffle_enabled": self.config.get('shuffle_enabled', False),
                "fletcher32_enabled": self.config.get('fletcher32_enabled', False),
                "batching_enabled": self.config.get('batching_enabled', True),
                "batch_size": self.config.get('batch_size', 32)
            }

            self.data_ready_for_saving.emit(data_package)
            # ### MODIFIED: Emit the 'labels' dictionary as the fourth argument ###
            self.scan_finished.emit(self.config['save_path'], final_rgb_preview, metadata, labels)
            
        except Exception as e:
            self.error_occurred.emit(f"Scan failed: {e}\n{traceback.format_exc()}")
        finally:
            if self.camera.is_connected and self.camera.cam.IsStreaming():
                self.camera.cam.EndAcquisition()

    def stop(self):
        self._is_interrupted = True

      
class AcquisitionDialog(QDialog):
    """
    Manages the data cube acquisition process with an advanced settings panel.
    """
    # ### MODIFIED: Added a 'dict' to match the thread's signal ###
    acquisition_complete = pyqtSignal(str, object, dict, dict)
    data_ready_for_saving = pyqtSignal(dict)
    scan_and_thread_finished = pyqtSignal(dict)
    settings_save_requested = pyqtSignal(dict)
    label_fields_save_requested = pyqtSignal(list)

    # ... (the rest of the AcquisitionDialog class is unchanged) ...
    def __init__(self, parent, camera_controller, stage_controller, config: dict, label_fields: list):
        super().__init__(parent)
        self.camera = camera_controller
        self.stage = stage_controller
        self.config = config
        self.scan_thread = None
        # ### NEW: Store label fields and widgets ###
        self.label_fields = label_fields
        self.label_input_widgets = {}

        self.setWindowTitle("Data Cube Acquisition")
        self.setMinimumSize(1000, 800)
        self.setModal(True)

        self._setup_ui()
        self._connect_signals()
        self._populate_settings_from_config()
        self._create_labeling_ui() # Populate the label widgets

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        settings_panel = QWidget()
        settings_panel.setMaximumWidth(400)
        settings_scroll = QScrollArea() # Add scroll area for many settings
        settings_scroll.setWidgetResizable(True)
        settings_scroll_widget = QWidget()
        settings_layout = QVBoxLayout(settings_scroll_widget)
        settings_scroll.setWidget(settings_scroll_widget)
        
        # ### NEW: Add the group box for labels ###
        self.labels_group = QGroupBox("Metadata Labels")
        self.labels_layout = QVBoxLayout(self.labels_group)
        self.label_widgets_layout = QFormLayout()
        manage_labels_button = QPushButton("Manage Label Fields...")
        self.labels_layout.addLayout(self.label_widgets_layout)
        self.labels_layout.addWidget(manage_labels_button, 0, Qt.AlignmentFlag.AlignRight)
        # Connect the button now that it's created
        manage_labels_button.clicked.connect(self._on_manage_labels)

        acq_group = QGroupBox("Acquisition & Save Settings")
        acq_layout = QFormLayout(acq_group)

        self.output_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.output_size_slider.setRange(10, 100)
        self.output_size_label = QLabel("100%")
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.output_size_slider); size_layout.addWidget(self.output_size_label)

        self.compression_combo = QComboBox()
        self.compression_combo.addItems(["None", "gzip", "lzf"])
        
        self.compression_level_slider = QSlider(Qt.Orientation.Horizontal)
        self.compression_level_slider.setRange(0, 9)
        self.compression_level_label = QLabel("4")
        comp_level_layout = QHBoxLayout()
        comp_level_layout.addWidget(self.compression_level_slider); comp_level_layout.addWidget(self.compression_level_label)
        self.compression_level_widget = QWidget()
        self.compression_level_widget.setLayout(comp_level_layout)

        self.shuffle_checkbox = QCheckBox("Enable Shuffle Filter (Improves Compression)")
        self.fletcher32_checkbox = QCheckBox("Enable Fletcher32 Checksum (Data Integrity)")
        
        self.batching_checkbox = QCheckBox("Enable Batch Writing for Compression")
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 256)
        self.batch_size_spinbox.setSuffix(" slices/batch")

        self.aspect_ratio_checkbox = QCheckBox("Maintain Aspect Ratio in RGB Preview")
        self.embed_rgb_checkbox = QCheckBox("Embed RGB Preview in HDF5 File")
        
        acq_layout.addRow("Output Size:", size_layout)
        acq_layout.addRow("Compression Algo:", self.compression_combo)
        acq_layout.addRow("GZIP Level:", self.compression_level_widget)
        acq_layout.addRow(self.shuffle_checkbox)
        acq_layout.addRow(self.fletcher32_checkbox)
        acq_layout.addRow(self.batching_checkbox)
        acq_layout.addRow("Batch Size:", self.batch_size_spinbox)
        acq_layout.addRow(self.aspect_ratio_checkbox)
        acq_layout.addRow(self.embed_rgb_checkbox)

        self.save_settings_button = QPushButton("Save These Settings to Config")
        
        # Assemble the settings panel
        settings_layout.addWidget(self.labels_group)
        settings_layout.addWidget(acq_group)
        settings_layout.addWidget(self.save_settings_button)
        settings_layout.addStretch()
        
        main_layout.addWidget(settings_scroll)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        save_loc_group = QGroupBox("Save Location")
        form_layout = QFormLayout(save_loc_group)
        dir_layout = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        dir_layout.addWidget(self.dir_edit); dir_layout.addWidget(self.browse_button)
        default_filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        self.filename_edit = QLineEdit(default_filename)
        form_layout.addRow("Save Directory:", dir_layout); form_layout.addRow("Filename:", self.filename_edit)
        
        preview_group = QGroupBox("Live RGB Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Scan will appear here..."); self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480); self.preview_label.setStyleSheet("background-color: #222; color: #888;")
        preview_layout.addWidget(self.preview_label)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar(); self.progress_bar.setTextVisible(True); self.progress_bar.setValue(0)
        self.status_log = QTextEdit(readOnly=True); self.status_log.setFixedHeight(100)
        progress_layout.addWidget(self.progress_bar); progress_layout.addWidget(self.status_log)
        
        self.button_box = QDialogButtonBox()
        self.go_to_start_button = self.button_box.addButton("Go to Start", QDialogButtonBox.ButtonRole.ActionRole)
        self.start_button = self.button_box.addButton("Start Scan", QDialogButtonBox.ButtonRole.ActionRole)
        self.cancel_button = self.button_box.addButton("Cancel Scan", QDialogButtonBox.ButtonRole.ActionRole)
        self.close_button = self.button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.cancel_button.setEnabled(False)
        
        right_layout.addWidget(save_loc_group); right_layout.addWidget(preview_group, 1)
        right_layout.addWidget(progress_group); right_layout.addWidget(self.button_box)
        
        main_layout.addWidget(right_panel, 1)

    def _connect_signals(self):
        self.browse_button.clicked.connect(self._browse_for_directory)
        self.go_to_start_button.clicked.connect(self._on_go_to_start)
        self.start_button.clicked.connect(self._on_start_scan)
        self.cancel_button.clicked.connect(self._on_cancel_scan)
        self.close_button.clicked.connect(self.reject)
        self.output_size_slider.valueChanged.connect(self._on_size_slider_changed)
        self.compression_level_slider.valueChanged.connect(self.compression_level_label.setNum)
        self.save_settings_button.clicked.connect(self._on_save_settings_clicked)
        self.compression_combo.currentTextChanged.connect(self._on_compression_algo_changed)
        self.batching_checkbox.toggled.connect(self.batch_size_spinbox.setEnabled)

    def _populate_settings_from_config(self):
        self.dir_edit.setText(self.config.get('data_cube_save_dir', os.path.expanduser("~")))
        
        self.output_size_slider.setValue(self.config.get('output_size_percentage', 100))
        self._on_size_slider_changed(self.output_size_slider.value())
        
        algo = self.config.get('compression_algo', 'gzip')
        if algo in [self.compression_combo.itemText(i) for i in range(self.compression_combo.count())]:
            self.compression_combo.setCurrentText(algo)
            
        self.compression_level_slider.setValue(self.config.get('compression_level', 4))
        self.compression_level_label.setNum(self.compression_level_slider.value())
        
        self.shuffle_checkbox.setChecked(self.config.get('shuffle_enabled', False))
        self.fletcher32_checkbox.setChecked(self.config.get('fletcher32_enabled', False))
        
        batching_on = self.config.get('batching_enabled', True)
        self.batching_checkbox.setChecked(batching_on)
        self.batch_size_spinbox.setValue(self.config.get('batch_size', 32))
        self.batch_size_spinbox.setEnabled(batching_on)

        self.aspect_ratio_checkbox.setChecked(self.config.get('maintain_aspect_ratio', True))
        self.embed_rgb_checkbox.setChecked(self.config.get('embed_rgb_preview', True))
        
        self._on_compression_algo_changed(self.compression_combo.currentText())

    def _get_settings_from_ui(self) -> dict:
        # ### NEW: Collect labels from the UI ###
        labels = {}
        for key, input_widget in self.label_input_widgets.items():
            labels[key] = input_widget.text()

        return {
            "output_size_percentage": self.output_size_slider.value(),
            "compression_algo": self.compression_combo.currentText(),
            "compression_level": self.compression_level_slider.value(),
            "shuffle_enabled": self.shuffle_checkbox.isChecked(),
            "fletcher32_enabled": self.fletcher32_checkbox.isChecked(),
            "batching_enabled": self.batching_checkbox.isChecked(),
            "batch_size": self.batch_size_spinbox.value(),
            "maintain_aspect_ratio": self.aspect_ratio_checkbox.isChecked(),
            "embed_rgb_preview": self.embed_rgb_checkbox.isChecked(),
            "data_cube_save_dir": self.dir_edit.text(),
            "labels": labels # Add the collected labels
        }

    # ### NEW METHOD: Dynamically creates the label input fields ###
    def _create_labeling_ui(self):
        # Clear existing widgets
        while self.label_widgets_layout.count():
            item = self.label_widgets_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.label_input_widgets.clear()
        
        # Create new widgets from the current field structure
        for field in self.label_fields:
            key = field.get("key")
            if not key: continue
            
            input_widget = QLineEdit()
            input_widget.setPlaceholderText(field.get("description", ""))
            
            self.label_widgets_layout.addRow(f"{key}:", input_widget)
            self.label_input_widgets[key] = input_widget

    # ### NEW METHOD: Handles managing label fields ###
    def _on_manage_labels(self):
        dialog = ManageLabelsDialog(self.label_fields, self)
        dialog.label_fields_updated.connect(self._on_label_fields_updated)
        dialog.exec()

    # ### NEW SLOT: Receives new field structure and updates UI ###
    @pyqtSlot(list)
    def _on_label_fields_updated(self, new_fields: list):
        self.log_message("Label fields structure updated.")
        self.label_fields = new_fields
        self._create_labeling_ui() # Rebuild the UI with the new fields
        self.label_fields_save_requested.emit(new_fields) # Ask main app to save
    
    @pyqtSlot(str)
    def _on_compression_algo_changed(self, text: str):
        is_gzip = (text == 'gzip')
        is_compressed = (text != 'None')
        self.compression_level_widget.setEnabled(is_gzip)
        self.shuffle_checkbox.setEnabled(is_compressed)
        self.fletcher32_checkbox.setEnabled(is_compressed)
        self.batching_checkbox.setEnabled(is_compressed)
        self.batch_size_spinbox.setEnabled(is_compressed and self.batching_checkbox.isChecked())

    def _on_save_settings_clicked(self):
        settings = self._get_settings_from_ui()
        # Don't save the actual label text to the config, just the settings
        settings.pop('labels', None)
        self.settings_save_requested.emit(settings)
        self.log_message("Request sent to save acquisition settings to config file.")

    def _on_size_slider_changed(self, value: int):
        has_calib = self.config.get("calibration_data") is not None
        if not has_calib:
            self.output_size_label.setText(f"{value}% (No Calib)")
            self.output_size_slider.setEnabled(False)
            return
        self.output_size_slider.setEnabled(True)
        output_dims = self.config["calibration_data"]["calibration_steps"]["output_dimensions"]
        max_w, max_h = output_dims['width'], output_dims['height']
        current_w = int(max_w * value / 100)
        current_h = int(max_h * value / 100)
        self.output_size_label.setText(f"{value}% ({current_w}x{current_h})")

    def _on_start_scan(self):
        self.config.update(self._get_settings_from_ui())
        save_dir = self.dir_edit.text()
        filename = self.filename_edit.text()
        if not save_dir or not filename:
            QMessageBox.warning(self, "Input Error", "Please specify a save directory and filename.")
            return
        if not filename.lower().endswith(".h5"):
            filename += ".h5"
            self.filename_edit.setText(filename)
        full_path = os.path.join(save_dir, filename)
        if os.path.exists(full_path):
            reply = QMessageBox.question(self, "File Exists", f"The file '{filename}' already exists. Overwrite?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.No: return
        self.config['save_path'] = full_path
        self.log_message("--- Starting Scan ---")
        self.start_button.setEnabled(False); self.go_to_start_button.setEnabled(False)
        self.cancel_button.setEnabled(True); self.close_button.setEnabled(False)
        self.progress_bar.setValue(0); self.preview_label.setText("Starting acquisition...")
        self.scan_thread = DataCubeAcquisitionThread(self.camera, self.stage, self.config, self)
        self.scan_thread.progress_update.connect(self._update_progress)
        self.scan_thread.rgb_preview_update.connect(self._update_live_preview)
        self.scan_thread.data_ready_for_saving.connect(self.data_ready_for_saving)
        self.scan_thread.scan_finished.connect(self.acquisition_complete)
        self.scan_thread.error_occurred.connect(self._on_scan_error)
        self.scan_thread.finished.connect(self._on_scan_thread_complete)
        self.scan_thread.finished.connect(self.scan_thread.deleteLater)
        self.scan_thread.start()

    def log_message(self, message: str):
        self.status_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def _browse_for_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.dir_edit.text())
        if directory: self.dir_edit.setText(directory)

    def _on_go_to_start(self):
        start_pos = self.config.get('start_position')
        speed = self.config.get('scan_speed', 200)
        if start_pos is not None:
            self.log_message(f"Moving to saved start position: {start_pos}...")
            self.stage.move_to(start_pos, speed)
        else:
            self.log_message("ERROR: Start position is not set.")
            QMessageBox.warning(self, "Missing Info", "Start position has not been set in the main window.")

    def _on_cancel_scan(self):
        if self.scan_thread and self.scan_thread.isRunning():
            self.log_message("Requesting scan cancellation...")
            self.scan_thread.stop()
            self.cancel_button.setEnabled(False)

    @pyqtSlot(int, int, str)
    def _update_progress(self, current: int, total: int, message: str):
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            self.progress_bar.setFormat(f"{current}/{total} - %p%")
        self.log_message(message)

    @pyqtSlot(object)
    def _update_live_preview(self, rgb_image: np.ndarray):
        if rgb_image is None or rgb_image.size == 0: return
        try:
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            self.log_message(f"Error updating live preview: {e}")

    @pyqtSlot(str)
    def _on_scan_error(self, message: str):
        self.log_message(f"ERROR: {message}")
        QMessageBox.critical(self, "Scan Error", message)

    @pyqtSlot()
    def _on_scan_thread_complete(self):
        self.start_button.setEnabled(True); self.go_to_start_button.setEnabled(True)
        self.cancel_button.setEnabled(False); self.close_button.setEnabled(True)
        self.scan_and_thread_finished.emit(self.config)
        self.scan_thread = None

    def closeEvent(self, event: QCloseEvent):
        if self.scan_thread is not None and self.scan_thread.isRunning():
            QMessageBox.warning(self, "Scan in Progress", "Please cancel the scan before closing the window.")
            event.ignore()
        else:
            event.accept()
    
class HsiControlApp(QMainWindow):
    """Main application window for the HSI System Controller."""

    def __init__(self):
        super().__init__()
        self.label_fields = [
            {"key": "Sample Name", "description": "e.g., 'SampleA_Test1', 'Red_Fabric_Swatch'"},
            {"key": "Notes", "description": "Any relevant notes about the scan..."}
        ]
        # ### ADDED: Dictionary to hold the dynamic label input widgets ###
        self.label_input_widgets = {}
        self.setWindowTitle("HSI System Control Panel")
        self.setGeometry(100, 100, 1400, 900)
        self.camera = CameraController(self)
        self.stage = StageController(self)
        self.last_frame = None
        self.last_scan_result = None
        self.last_scan_metadata = None
        self.current_stage_position = None
        self.scan_start_pos = None
        self.scan_end_pos = None
        self.calibration_data = None
        self.current_calib_file_path = "None"
        self.data_cube_save_dir = os.path.expanduser("~")
        self.last_h5_path = None
        self.fps_timer = QTimer(self)
        self.frame_count = 0
        self.fps_start_time = 0
        self.current_fps = 0.0
        self.output_size_percentage = 100
        self.compression_algo = 'None' 
        self.compression_level = 4      
        self.shuffle_enabled = False
        self.fletcher32_enabled = False
        self.batching_enabled = True
        self.batch_size = 32

        self.data_cube_for_cropping = None
        self.crop_mode = "Bounding Box"
        self.active_crop_rect_normalized = QRectF(0.0, 0.0, 1.0, 1.0) 
        self.remember_crop_settings = False

        self.save_thread = None
        self.wizard_window = None
        self.camera_test_window = None

        self._setup_ui()
        self._connect_signals()
        self._load_config()
        # ### ADDED: Create the initial labeling panel from the loaded config ###
        self._update_labeling_panel() 
        self._load_default_calibration()
        self._update_ui_state()
        
    def _setup_ui(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        self.open_viewer_action = QAction("&Preview Data Cube...", self)
        file_menu.addAction(self.open_viewer_action)
        self.load_for_editing_action = QAction("&Load Data Cube for Editing...", self)
        file_menu.addAction(self.load_for_editing_action)
        calibration_menu = menu_bar.addMenu("&Calibration")
        self.run_wizard_action = QAction("&Run Calibration Wizard...", self)
        calibration_menu.addAction(self.run_wizard_action)
        tools_menu = menu_bar.addMenu("&Tools")
        self.launch_slice_viewer_action = QAction("Advanced Slice &Analyzer...", self)
        tools_menu.addAction(self.launch_slice_viewer_action)
        tools_menu.addSeparator()
        self.launch_camera_app_action = QAction("&Launch Dedicated Camera App...", self)
        tools_menu.addAction(self.launch_camera_app_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setMinimumWidth(450)
        controls_container_widget = QWidget()
        controls_layout = QVBoxLayout(controls_container_widget)
        left_scroll_area.setWidget(controls_container_widget)
        main_layout.addWidget(left_scroll_area)
        self._setup_connection_group(controls_layout)
        self._setup_stage_status_group(controls_layout)
        self._setup_calibration_group(controls_layout)
        self._setup_stage_control_group(controls_layout)
        self._setup_scan_range_group(controls_layout)
        self._setup_camera_control_group(controls_layout)
        self._setup_log_group(controls_layout)
        controls_layout.addStretch()
        
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, 1)

        overlays_group = QGroupBox("Live View Overlays")
        overlays_layout = QHBoxLayout(overlays_group)
        self.grid_checkbox = QCheckBox("Grid")
        self.center_line_checkbox = QCheckBox("Center Line")
        self.spectral_lines_checkbox = QCheckBox("Spectral Lines")
        self.wavelength_ruler_checkbox = QCheckBox("Wavelength Ruler")
        overlays_layout.addWidget(self.grid_checkbox)
        overlays_layout.addWidget(self.center_line_checkbox)
        overlays_layout.addWidget(self.spectral_lines_checkbox)
        overlays_layout.addWidget(self.wavelength_ruler_checkbox)
        right_panel_layout.addWidget(overlays_group)

        image_and_tools_layout = QHBoxLayout()

        self.image_label = InteractiveCropLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #333; color: white;")
        self.image_label.setMinimumSize(640, 480)
        image_and_tools_layout.addWidget(self.image_label, 2)

        tools_scroll_area = QScrollArea()
        tools_scroll_area.setWidgetResizable(True)
        tools_scroll_area.setMaximumWidth(400)
        
        tools_container = QWidget()
        tools_layout = QVBoxLayout(tools_container)
        
        # ### RESTORED: Labeling Group is added back to the main UI ###
        self.labeling_group = QGroupBox("Post-Scan Labeling")
        self.labeling_layout = QVBoxLayout(self.labeling_group)
        label_button_layout = QHBoxLayout()
        manage_labels_button = QPushButton("Manage Label Fields...")
        self.save_labels_button = QPushButton("Save Labels to Original HDF5 File")
        label_button_layout.addWidget(manage_labels_button)
        label_button_layout.addStretch()
        label_button_layout.addWidget(self.save_labels_button)
        self.label_widgets_layout = QFormLayout()
        self.labeling_layout.addLayout(self.label_widgets_layout)
        self.labeling_layout.addLayout(label_button_layout)
        self.labeling_group.setVisible(False)
        manage_labels_button.clicked.connect(self._on_manage_labels)
        tools_layout.addWidget(self.labeling_group)

        # Cropping Group
        self.cropping_group = QGroupBox("Post-Scan Cropping")
        self._setup_cropping_group(self.cropping_group)
        self.cropping_group.setVisible(False)
        tools_layout.addWidget(self.cropping_group)
        
        self.save_progress_group = QGroupBox("Saving Progress")
        save_progress_layout = QVBoxLayout(self.save_progress_group)
        self.save_status_label = QLabel("Initializing save...")
        self.save_progress_bar = QProgressBar()
        save_progress_layout.addWidget(self.save_status_label)
        save_progress_layout.addWidget(self.save_progress_bar)
        self.save_progress_group.setVisible(False) 
        tools_layout.addWidget(self.save_progress_group)
        
        tools_layout.addStretch()
        tools_scroll_area.setWidget(tools_container)
        image_and_tools_layout.addWidget(tools_scroll_area, 2)

        right_panel_layout.addLayout(image_and_tools_layout)
        
        self.show_scan_result_button = QPushButton("Clear Scan Result and Resume Live View")
        self.show_scan_result_button.setVisible(False)
        right_panel_layout.addWidget(self.show_scan_result_button)

        self.statusBar = self.statusBar()
        self.fps_status_label = QLabel("FPS: N/A")
        self.img_size_status_label = QLabel("Size: N/A")
        self.statusBar.addPermanentWidget(self.fps_status_label)
        self.statusBar.addPermanentWidget(self.img_size_status_label)
        

    def _setup_cropping_group(self, group_box: QGroupBox):
        """Creates the new, advanced cropping UI elements."""
        layout = QVBoxLayout(group_box)
        
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Crop Mode:"))
        self.crop_mode_box_radio = QRadioButton("Bounding Box")
        self.crop_mode_box_radio.setChecked(True)
        self.crop_mode_slider_radio = QRadioButton("Sliders")
        mode_layout.addWidget(self.crop_mode_box_radio)
        mode_layout.addWidget(self.crop_mode_slider_radio)
        mode_layout.addStretch()
        
        self.sliders_panel = QWidget()
        sliders_layout = QFormLayout(self.sliders_panel)
        sliders_layout.setContentsMargins(0, 5, 0, 5)

        self.crop_slider_left = QSlider(Qt.Orientation.Horizontal)
        self.crop_slider_right = QSlider(Qt.Orientation.Horizontal)
        self.crop_slider_top = QSlider(Qt.Orientation.Horizontal)
        self.crop_slider_bottom = QSlider(Qt.Orientation.Horizontal)
        
        self.crop_label_left = QLabel("0%")
        self.crop_label_right = QLabel("100%")
        self.crop_label_top = QLabel("0%")
        self.crop_label_bottom = QLabel("100%")
        
        for slider in [self.crop_slider_left, self.crop_slider_right, self.crop_slider_top, self.crop_slider_bottom]:
            slider.setRange(0, 1000)

        self.crop_slider_left.setValue(0)
        self.crop_slider_right.setValue(1000)
        self.crop_slider_top.setValue(0)
        self.crop_slider_bottom.setValue(1000)

        def create_slider_row(label_widget, slider_widget):
            row = QHBoxLayout()
            row.addWidget(slider_widget)
            row.addWidget(label_widget)
            return row

        sliders_layout.addRow("Left:", create_slider_row(self.crop_label_left, self.crop_slider_left))
        sliders_layout.addRow("Right:", create_slider_row(self.crop_label_right, self.crop_slider_right))
        sliders_layout.addRow("Top:", create_slider_row(self.crop_label_top, self.crop_slider_top))
        sliders_layout.addRow("Bottom:", create_slider_row(self.crop_label_bottom, self.crop_slider_bottom))
        
        action_layout = QHBoxLayout()
        self.remember_crop_checkbox = QCheckBox("Remember this crop")
        self.reset_crop_button = QPushButton("Reset Crop")
        self.crop_save_button = QPushButton("Crop & Save As New HDF5...")
        action_layout.addWidget(self.remember_crop_checkbox)
        action_layout.addStretch()
        action_layout.addWidget(self.reset_crop_button)
        action_layout.addWidget(self.crop_save_button)

        layout.addLayout(mode_layout)
        layout.addWidget(self.sliders_panel)
        layout.addLayout(action_layout)

        self.sliders_panel.setEnabled(False)

    def _setup_connection_group(self, parent_layout):
        conn_group = QGroupBox("System Connection & Configuration")
        conn_layout = QHBoxLayout(conn_group)
        self.connect_all_button = QPushButton("Connect All Devices")
        self.disconnect_all_button = QPushButton("Disconnect All")
        self.save_config_button = QPushButton("Save Current Config")
        self.save_config_button.setToolTip("Saves ALL current settings (scan, camera, etc.) to the config file.")
        conn_layout.addWidget(self.connect_all_button)
        conn_layout.addWidget(self.disconnect_all_button)
        conn_layout.addWidget(self.save_config_button)
        parent_layout.addWidget(conn_group)

    def _setup_stage_status_group(self, parent_layout):
        stage_status_group = QGroupBox("Stage Status")
        stage_status_layout = QFormLayout(stage_status_group)
        self.stage_conn_status_label = QLabel("Disconnected")
        self.stage_conn_status_label.setStyleSheet("color: red; font-weight: bold;")
        self.stage_homing_status_label = QLabel("Not Homed")
        self.stage_homing_status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.stage_current_pos_label = QLabel("N/A")
        self.stage_current_pos_label.setStyleSheet("font-weight: bold;")
        stage_status_layout.addRow("Connection:", self.stage_conn_status_label)
        stage_status_layout.addRow("System Ready:", self.stage_homing_status_label)
        stage_status_layout.addRow("Current Position:", self.stage_current_pos_label)
        parent_layout.addWidget(stage_status_group)

    def _setup_calibration_group(self, parent_layout):
        calibration_group = QGroupBox("System Calibration & Visualization")
        calib_layout = QFormLayout(calibration_group)
        self.load_calib_button = QPushButton("Load Master Calibration File...")
        self.calib_status_label = QLabel("No file loaded.")
        self.calib_status_label.setStyleSheet("font-style: italic;")
        self.apply_geo_correction_checkbox = QCheckBox("Apply Geometric Correction to Live View")
        self.mm_per_step_spinbox = QDoubleSpinBox()
        self.mm_per_step_spinbox.setDecimals(4); self.mm_per_step_spinbox.setRange(0.0001, 10.0); self.mm_per_step_spinbox.setSingleStep(0.001); self.mm_per_step_spinbox.setSuffix(" mm/step")
        self.camera_fov_width_mm_spinbox = QDoubleSpinBox()
        self.camera_fov_width_mm_spinbox.setDecimals(2); self.camera_fov_width_mm_spinbox.setRange(0.1, 200.0); self.camera_fov_width_mm_spinbox.setSingleStep(0.1); self.camera_fov_width_mm_spinbox.setSuffix(" mm")
        self.scan_fps_spinbox = QSpinBox()
        self.scan_fps_spinbox.setRange(1, 200); self.scan_fps_spinbox.setSuffix(" FPS")
        self.scan_duration_spinbox = QDoubleSpinBox()
        self.scan_duration_spinbox.setDecimals(1); self.scan_duration_spinbox.setRange(0.1, 3600.0); self.scan_duration_spinbox.setValue(10.0); self.scan_duration_spinbox.setSuffix(" s")
        self.r_wavelength_spinbox = QDoubleSpinBox()
        self.r_wavelength_spinbox.setRange(380.0, 1000.0); self.r_wavelength_spinbox.setValue(640.0); self.r_wavelength_spinbox.setSuffix(" nm")
        self.g_wavelength_spinbox = QDoubleSpinBox()
        self.g_wavelength_spinbox.setRange(380.0, 1000.0); self.g_wavelength_spinbox.setValue(550.0); self.g_wavelength_spinbox.setSuffix(" nm")
        self.b_wavelength_spinbox = QDoubleSpinBox()
        self.b_wavelength_spinbox.setRange(380.0, 1000.0); self.b_wavelength_spinbox.setValue(460.0); self.b_wavelength_spinbox.setSuffix(" nm")
        self.sampling_thickness_spinbox = QSpinBox()
        self.sampling_thickness_spinbox.setRange(1, 21); self.sampling_thickness_spinbox.setSingleStep(2); self.sampling_thickness_spinbox.setValue(5); self.sampling_thickness_spinbox.setSuffix(" px")
        calib_layout.addRow(self.load_calib_button)
        calib_layout.addRow("Status:", self.calib_status_label)
        calib_layout.addRow(self.apply_geo_correction_checkbox)
        calib_layout.addRow("Motion (mm per Step):", self.mm_per_step_spinbox)
        calib_layout.addRow("Camera FoV Width:", self.camera_fov_width_mm_spinbox)
        calib_layout.addRow("Target Scan FPS:", self.scan_fps_spinbox)
        calib_layout.addRow("Scan Duration:", self.scan_duration_spinbox)
        calib_layout.addRow("Red Wavelength:", self.r_wavelength_spinbox)
        calib_layout.addRow("Green Wavelength:", self.g_wavelength_spinbox)
        calib_layout.addRow("Blue Wavelength:", self.b_wavelength_spinbox)
        calib_layout.addRow("Scan Sampling Thickness:", self.sampling_thickness_spinbox)
        parent_layout.addWidget(calibration_group)

    def _setup_stage_control_group(self, parent_layout):
        self.stage_control_group = QGroupBox("Manual Stage Control")
        stage_control_layout = QFormLayout(self.stage_control_group)
        self.stage_min_limit_spinbox = QSpinBox()
        self.stage_min_limit_spinbox.setRange(0, 10000); self.stage_min_limit_spinbox.setValue(10); self.stage_min_limit_spinbox.setSuffix(" steps")
        self.stage_max_limit_spinbox = QSpinBox()
        self.stage_max_limit_spinbox.setRange(0, 10000); self.stage_max_limit_spinbox.setValue(250); self.stage_max_limit_spinbox.setSuffix(" steps")
        self.stage_pos_spinbox = QSpinBox()
        self.stage_pos_spinbox.setSuffix(" steps")
        self.stage_speed_spinbox = QSpinBox()
        self.stage_speed_spinbox.setRange(50, 1000); self.stage_speed_spinbox.setValue(200); self.stage_speed_spinbox.setSuffix(" steps/s")
        self.move_stage_button = QPushButton("Move to Position")
        stage_control_layout.addRow("Min Position Limit:", self.stage_min_limit_spinbox)
        stage_control_layout.addRow("Max Position Limit:", self.stage_max_limit_spinbox)
        stage_control_layout.addRow("Target Position:", self.stage_pos_spinbox)
        stage_control_layout.addRow("Speed:", self.stage_speed_spinbox)
        stage_control_layout.addRow(self.move_stage_button)
        parent_layout.addWidget(self.stage_control_group)

    def _setup_scan_range_group(self, parent_layout):
        scan_config_group = QGroupBox("Scan Configuration")
        scan_config_layout = QFormLayout(scan_config_group)
        
        workflow_label = QLabel("<b>Workflow:</b> 1. Set Start/End → 2. Acquire Data Cube")
        workflow_label.setWordWrap(True)
        self.scan_start_label = QLabel("Not Set")
        self.scan_end_label = QLabel("Not Set")
        set_pos_layout = QHBoxLayout()
        self.set_start_button = QPushButton("Set Current as Start")
        self.set_end_button = QPushButton("Set Current as End")
        set_pos_layout.addWidget(self.set_start_button)
        set_pos_layout.addWidget(self.set_end_button)
        self.scan_speed_spinbox = QSpinBox()
        self.scan_speed_spinbox.setRange(50, 1000); self.scan_speed_spinbox.setValue(100); self.scan_speed_spinbox.setSuffix(" steps/s")
        
        self.return_to_start_checkbox = QCheckBox("Return to Start Position After Scan")
        self.post_scan_wait_spinbox = QSpinBox()
        self.post_scan_wait_spinbox.setRange(0, 300) 
        self.post_scan_wait_spinbox.setValue(5)
        self.post_scan_wait_spinbox.setSuffix(" s")
        self.post_scan_wait_spinbox.setEnabled(False)

        self.acquire_button = QPushButton("Acquire Data Cube...")
        self.acquire_button.setToolTip("Opens the acquisition dialog. Make sure the stage is at the Start Position first!")
        
        scan_config_layout.addRow(workflow_label)
        scan_config_layout.addRow("Start Position:", self.scan_start_label)
        scan_config_layout.addRow("End Position:", self.scan_end_label)
        scan_config_layout.addRow(set_pos_layout)
        scan_config_layout.addRow("Scan Speed:", self.scan_speed_spinbox)
        
        scan_config_layout.addRow(self.return_to_start_checkbox)
        scan_config_layout.addRow("Wait Time Before Return:", self.post_scan_wait_spinbox)
        
        scan_config_layout.addRow(self.acquire_button)
        parent_layout.addWidget(scan_config_group)

    def _setup_camera_control_group(self, parent_layout):
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QFormLayout(camera_group)
        self.exposure_spinbox = QDoubleSpinBox()
        self.exposure_spinbox.setRange(1.0, 1000000.0); self.exposure_spinbox.setValue(8000); self.exposure_spinbox.setSuffix(" µs"); self.exposure_spinbox.setSingleStep(100); self.exposure_spinbox.setDecimals(0)
        
        self.start_live_view_button = QPushButton("Start Live View")
        self.stop_live_view_button = QPushButton("Stop Live View")
        camera_layout.addRow("Exposure Time:", self.exposure_spinbox)
        camera_layout.addRow(self.start_live_view_button, self.stop_live_view_button)
        parent_layout.addWidget(camera_group)

    def _setup_log_group(self, parent_layout):
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit(readOnly=True)
        log_layout.addWidget(self.log_text_edit)
        parent_layout.addWidget(log_group)

    def _connect_signals(self):
        self.open_viewer_action.triggered.connect(self._on_open_viewer)
        self.load_for_editing_action.triggered.connect(self._on_load_for_editing)
        self.run_wizard_action.triggered.connect(self._on_run_calibration_wizard)
        self.launch_slice_viewer_action.triggered.connect(self._on_launch_slice_viewer)
        self.launch_camera_app_action.triggered.connect(self._on_launch_dedicated_camera_app)
        self.connect_all_button.clicked.connect(self._on_connect_all)
        self.disconnect_all_button.clicked.connect(self._on_disconnect_all)
        self.save_config_button.clicked.connect(self._on_save_config)
        self.load_calib_button.clicked.connect(self._on_load_calibration)
        self.acquire_button.clicked.connect(self._on_acquire_data_cube)
        self.camera.status_update.connect(self.log_message)
        self.camera.connection_lost.connect(self._on_camera_connection_lost)
        self.camera.new_live_frame.connect(self._update_image_display)
        self.camera.exposure_time_updated.connect(self._on_exposure_time_updated)
        self.start_live_view_button.clicked.connect(self._on_start_live_view)
        self.stop_live_view_button.clicked.connect(self._on_stop_live_view)
        self.exposure_spinbox.valueChanged.connect(self.camera.set_exposure_time)
        self.stage.status_update.connect(self.log_message)
        self.stage.homing_complete.connect(self._on_homing_complete)
        self.stage.connection_lost.connect(self._on_stage_connection_lost)
        self.move_stage_button.clicked.connect(self._on_move_stage)
        self.set_start_button.clicked.connect(self._on_set_start_pos)
        self.set_end_button.clicked.connect(self._on_set_end_pos)
        self.stage_min_limit_spinbox.valueChanged.connect(self._on_stage_limits_changed)
        self.stage_max_limit_spinbox.valueChanged.connect(self._on_stage_limits_changed)
        self.show_scan_result_button.clicked.connect(self._on_clear_scan_result)
        # ### RESTORED: Signal for the save labels button ###
        self.save_labels_button.clicked.connect(self._on_save_labels)
        for widget in [self.grid_checkbox, self.center_line_checkbox, self.spectral_lines_checkbox,
                       self.wavelength_ruler_checkbox, self.apply_geo_correction_checkbox]:
            widget.clicked.connect(self._trigger_redraw)
        for widget in [self.r_wavelength_spinbox, self.g_wavelength_spinbox, self.b_wavelength_spinbox]:
            widget.valueChanged.connect(self._trigger_redraw)
        self.fps_timer.timeout.connect(self._update_fps)
        
        self.image_label.crop_rect_changed.connect(self._on_crop_box_drawn)
        self.crop_mode_box_radio.toggled.connect(self._on_crop_mode_changed)
        self.crop_slider_left.valueChanged.connect(self._on_crop_slider_changed)
        self.crop_slider_right.valueChanged.connect(self._on_crop_slider_changed)
        self.crop_slider_top.valueChanged.connect(self._on_crop_slider_changed)
        self.crop_slider_bottom.valueChanged.connect(self._on_crop_slider_changed)
        self.reset_crop_button.clicked.connect(self._on_reset_crop)
        self.crop_save_button.clicked.connect(self._on_crop_and_save)
        self.remember_crop_checkbox.toggled.connect(self._on_remember_crop_toggled)

    def _on_open_viewer(self):
        """Opens a file dialog and launches the DataCubeViewerDialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data Cube",
            self.data_cube_save_dir,
            "HDF5 Files (*.h5 *.hdf5)"
        )
        if filepath:
            try:
                viewer = DataCubeViewerDialog(filepath, self)
                viewer.exec()
            except Exception as e:
                self.log_message(f"ERROR: Failed to open viewer dialog: {e}")
                QMessageBox.critical(self, "Viewer Error", f"Could not open the data cube viewer:\n\n{e}")
    
    def _on_load_for_editing(self):
        """
        Opens a file dialog to load an existing HDF5 data cube, populating the UI
        with its data and enabling the post-processing panels.
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data Cube for Editing",
            self.data_cube_save_dir,
            "HDF5 Files (*.h5 *.hdf5)"
        )

        if not filepath: return

        try:
            self.log_message(f"Loading data cube for editing: {os.path.basename(filepath)}")
            data_cube, metadata, labels, rgb_preview = file_io.load_h5(filepath)

            if rgb_preview is None:
                QMessageBox.warning(self, "Load Warning", "The selected HDF5 file does not contain an embedded RGB preview and cannot be edited in this view.")
                return

            self._on_clear_scan_result()

            self.last_h5_path = filepath
            self.last_scan_result = rgb_preview
            self.last_scan_metadata = metadata
            self.data_cube_for_cropping = data_cube

            # ### RESTORED: Populate label widgets with loaded data ###
            if labels:
                for key, widget in self.label_input_widgets.items():
                    widget.setText(labels.get(key, ""))
            
            scan_params = metadata.get("scan_parameters", {})
            self.scan_speed_spinbox.setValue(scan_params.get("scan_speed", 100))
            self.mm_per_step_spinbox.setValue(scan_params.get("mm_per_step", 0.01))
            self.camera_fov_width_mm_spinbox.setValue(scan_params.get("camera_fov_width_mm", 10.0))
            self.scan_fps_spinbox.setValue(scan_params.get("scan_fps", 30))
            self.scan_duration_spinbox.setValue(scan_params.get("scan_duration_seconds", 10.0))
            self.r_wavelength_spinbox.setValue(scan_params.get("r_wavelength", 640.0))
            self.g_wavelength_spinbox.setValue(scan_params.get("g_wavelength", 550.0))
            self.b_wavelength_spinbox.setValue(scan_params.get("b_wavelength", 460.0))
            self.sampling_thickness_spinbox.setValue(scan_params.get("sampling_thickness", 5))
            self.output_size_percentage = scan_params.get("output_size_percentage", 100)
            self.exposure_spinbox.setValue(scan_params.get("exposure_time_us", 8000))
            self._deactivate_spectral_overlays()
            
            self._update_image_display()
            self.show_scan_result_button.setVisible(True)
            # ### RESTORED: Show the labeling and cropping groups ###
            self.labeling_group.setVisible(True)
            self.cropping_group.setVisible(True)
            self.log_message(f"Successfully loaded '{os.path.basename(filepath)}'. Ready for labeling or cropping.")

        except Exception as e:
            self.log_message(f"ERROR: Failed to load data cube for editing: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Load Error", f"Could not load or parse the selected HDF5 file:\n\n{e}")
        
    def _on_run_calibration_wizard(self):
        """Launches the calibration wizard GUI in the current process."""
        try:
            if self.wizard_window is None or not self.wizard_window.isVisible():
                self.log_message("Launching Calibration Wizard window...")
                self.wizard_window = CalibrationWizard(project_dir=_CURRENT_DIR)
                self.wizard_window.show()
            else:
                self.log_message("Calibration Wizard is already open. Activating window.")
                self.wizard_window.activateWindow()
                self.wizard_window.raise_()

        except Exception as e:
            error_msg = f"Failed to launch the Calibration Wizard window.\n\nError: {e}"
            self.log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)
    
    def _on_launch_dedicated_camera_app(self):
        """Launches the dedicated camera test app in the current process."""
        try:
            if self.camera_test_window is None or not self.camera_test_window.isVisible():
                self.log_message("Launching Dedicated Camera App window...")
                self.camera_test_window = CameraTestApp(file_dir=_CURRENT_DIR)
                self.camera_test_window.show()
            else:
                self.log_message("Dedicated Camera App is already open. Activating window.")
                self.camera_test_window.activateWindow()
                self.camera_test_window.raise_()

        except Exception as e:
            error_msg = f"Failed to launch the Dedicated Camera App window.\n\nError: {e}"
            self.log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Launch Error", error_msg)
            
    def _update_ui_state(self):
        cam_connected = self.camera.is_connected
        is_acquiring = self.camera.is_acquiring
        stage_connected = self.stage.is_connected
        stage_homed = self.stage.is_homed
        has_calibration = self.calibration_data is not None
        can_scan = all([cam_connected, stage_homed, has_calibration, self.scan_start_pos is not None, self.scan_end_pos is not None])
        self.start_live_view_button.setEnabled(cam_connected and not is_acquiring)
        self.stop_live_view_button.setEnabled(is_acquiring)
        self.exposure_spinbox.setEnabled(cam_connected)
        self.stage_control_group.setEnabled(stage_connected and stage_homed)
        self.set_start_button.setEnabled(stage_connected and stage_homed and self.current_stage_position is not None)
        self.set_end_button.setEnabled(stage_connected and stage_homed and self.current_stage_position is not None)
        self.acquire_button.setEnabled(can_scan and not is_acquiring)
        self.connect_all_button.setEnabled(not cam_connected and not stage_connected)
        self.disconnect_all_button.setEnabled(cam_connected or stage_connected)
        self.apply_geo_correction_checkbox.setEnabled(has_calibration)

        self.wavelength_ruler_checkbox.setEnabled(is_acquiring and has_calibration)
        self.spectral_lines_checkbox.setEnabled(is_acquiring and has_calibration)

        self.stage_conn_status_label.setText("Connected" if stage_connected else "Disconnected")
        self.stage_conn_status_label.setStyleSheet(f"color: {'green' if stage_connected else 'red'}; font-weight: bold;")
        self.stage_homing_status_label.setText("Ready (Homed)" if stage_homed else "Not Homed")
        self.stage_homing_status_label.setStyleSheet(f"color: {'green' if stage_homed else 'orange'}; font-weight: bold;")

    def _get_current_config(self) -> dict:
        config = {
            "start_position": self.scan_start_pos, "end_position": self.scan_end_pos,
            "scan_speed": self.scan_speed_spinbox.value(),
            "mm_per_step": self.mm_per_step_spinbox.value(),
            "camera_fov_width_mm": self.camera_fov_width_mm_spinbox.value(),
            "scan_fps": self.scan_fps_spinbox.value(),
            "scan_duration_seconds": self.scan_duration_spinbox.value(),
            "r_wavelength": self.r_wavelength_spinbox.value(),
            "g_wavelength": self.g_wavelength_spinbox.value(),
            "b_wavelength": self.b_wavelength_spinbox.value(),
            "sampling_thickness": self.sampling_thickness_spinbox.value(),
            "stage_min_limit": self.stage_min_limit_spinbox.value(),
            "stage_max_limit": self.stage_max_limit_spinbox.value(),
            "data_cube_save_dir": self.data_cube_save_dir,
            "output_size_percentage": self.output_size_percentage,
            "exposure_time_us": self.exposure_spinbox.value(),
            "compression_algo": self.compression_algo,
            "compression_level": self.compression_level,
            "shuffle_enabled": self.shuffle_enabled,
            "fletcher32_enabled": self.fletcher32_enabled,
            "batching_enabled": self.batching_enabled,
            "batch_size": self.batch_size,
            "custom_label_fields": self.label_fields,
            "return_to_start_after_scan": self.return_to_start_checkbox.isChecked(),
            "post_scan_wait_seconds": self.post_scan_wait_spinbox.value(),
            "remember_crop_settings": self.remember_crop_settings
        }
        if self.remember_crop_settings:
            rect = self.active_crop_rect_normalized
            config["saved_crop_settings"] = {
                "mode": self.crop_mode,
                "rect_normalized": [rect.x(), rect.y(), rect.width(), rect.height()]
            }
        return config

    def _load_config(self):
        config = {}
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                self.log_message(f"Loaded configuration from {CONFIG_FILE}")
            else:
                self.log_message(f"Configuration file '{CONFIG_FILE}' not found. Using default values.")
        except json.JSONDecodeError as e:
            self.log_message(f"ERROR: Could not parse '{CONFIG_FILE}'. It may be corrupt. {e}")
        except Exception as e:
            self.log_message(f"ERROR: Failed to read '{CONFIG_FILE}'. {e}")

        # Populate state and UI from the config dictionary
        self.scan_start_pos = config.get('start_position')
        self.scan_end_pos = config.get('end_position')
        self.scan_start_label.setText(str(self.scan_start_pos) if self.scan_start_pos is not None else "Not Set")
        self.scan_end_label.setText(str(self.scan_end_pos) if self.scan_end_pos is not None else "Not Set")
        
        self.scan_speed_spinbox.setValue(config.get('scan_speed', 100))
        self.mm_per_step_spinbox.setValue(config.get('mm_per_step', 0.01))
        self.camera_fov_width_mm_spinbox.setValue(config.get('camera_fov_width_mm', 10.0))
        self.scan_fps_spinbox.setValue(config.get('scan_fps', 30))
        self.scan_duration_spinbox.setValue(config.get('scan_duration_seconds', 10.0))
        self.r_wavelength_spinbox.setValue(config.get('r_wavelength', 640.0))
        self.g_wavelength_spinbox.setValue(config.get('g_wavelength', 550.0))
        self.b_wavelength_spinbox.setValue(config.get('b_wavelength', 460.0))
        self.sampling_thickness_spinbox.setValue(config.get('sampling_thickness', 5))
        self.stage_min_limit_spinbox.setValue(config.get('stage_min_limit', 10))
        self.stage_max_limit_spinbox.setValue(config.get('stage_max_limit', 250))
        self.data_cube_save_dir = config.get('data_cube_save_dir', os.path.expanduser("~"))
        self.output_size_percentage = config.get('output_size_percentage', 100)
        self.exposure_spinbox.setValue(config.get('exposure_time_us', 8000))
        self.compression_algo = config.get('compression_algo', 'None')
        self.compression_level = config.get('compression_level', 4)
        self.shuffle_enabled = config.get('shuffle_enabled', False)
        self.fletcher32_enabled = config.get('fletcher32_enabled', False)
        self.batching_enabled = config.get('batching_enabled', True)
        self.batch_size = config.get('batch_size', 32)
        
        return_enabled = config.get("return_to_start_after_scan", False)
        self.return_to_start_checkbox.setChecked(return_enabled)
        self.post_scan_wait_spinbox.setValue(config.get("post_scan_wait_seconds", 5))
        self.post_scan_wait_spinbox.setEnabled(return_enabled)

        default_fields = [
            {"key": "Sample Name", "description": "e.g., 'SampleA_Test1'"},
            {"key": "Notes", "description": "Any relevant notes..."}
        ]
        self.label_fields = config.get("custom_label_fields", default_fields)
        
        self.remember_crop_settings = config.get("remember_crop_settings", False)
        self.remember_crop_checkbox.setChecked(self.remember_crop_settings)
        if self.remember_crop_settings and "saved_crop_settings" in config:
            crop_cfg = config["saved_crop_settings"]
            self.crop_mode = crop_cfg.get("mode", "Bounding Box")
            rect_list = crop_cfg.get("rect_normalized", [0, 0, 1, 1])
            self.active_crop_rect_normalized = QRectF(*rect_list)
        else:
            self._on_reset_crop() 
            
        self._sync_crop_ui_from_state()
        self._on_stage_limits_changed()
        self._update_ui_state()

    def _load_and_validate_calibration_file(self, path: str):
        try:
            with open(path, 'r') as f: data = json.load(f)
            calib_steps = data.get("calibration_steps", {})
            required_keys = ["straightening", "cropping", "spectral", "output_dimensions"]
            if not all(key in calib_steps for key in required_keys):
                raise ValueError(f"JSON is missing one or more required keys: {required_keys}")
            if not all(key in calib_steps["output_dimensions"] for key in ["width", "height"]):
                raise ValueError("output_dimensions must contain 'width' and 'height'.")
            self.calibration_data = data
            self.current_calib_file_path = os.path.basename(path)
            self.calib_status_label.setText(self.current_calib_file_path)
            self.calib_status_label.setStyleSheet("font-style: normal; color: green;")
            self.log_message(f"Successfully loaded calibration file: {path}")
        except Exception as e:
            self.calibration_data = None
            self.current_calib_file_path = "None"
            self.calib_status_label.setText("Load failed. Invalid file.")
            self.calib_status_label.setStyleSheet("font-style: italic; color: red;")
            QMessageBox.critical(self, "Load Error", f"Could not load or validate the calibration file.\n\nError: {e}")
        finally:
            self._update_ui_state()
            self._trigger_redraw()

    def _on_size_slider_changed(self, value: int):
        if not self.calibration_data:
            return
        output_dims = self.calibration_data["calibration_steps"]["output_dimensions"]
        max_w, max_h = output_dims['width'], output_dims['height']
        current_w = int(max_w * value / 100)
        current_h = int(max_h * value / 100)
        self.output_size_label.setText(f"{value}% ({current_w}x{current_h})")

    @pyqtSlot(str, object, dict, dict)
    def _on_acquisition_complete(self, h5_path: str, rgb_preview: np.ndarray, metadata: dict, labels: dict):
        self.log_message(f"Acquisition complete. Result from {os.path.basename(h5_path)} received.")
        self.last_h5_path = h5_path
        self.last_scan_result = rgb_preview
        self.last_scan_metadata = metadata
        
        # ### NEW: Populate the label widgets with the data from the scan ###
        if labels:
            for key, widget in self.label_input_widgets.items():
                widget.setText(labels.get(key, ""))

        self.show_scan_result_button.setVisible(True)
        self.labeling_group.setVisible(True)
        self.cropping_group.setVisible(True)
        self._deactivate_spectral_overlays()
        self._update_image_display()

    def _on_clear_scan_result(self):
        self.last_scan_result = None
        self.last_h5_path = None
        self.last_scan_metadata = None
        self.data_cube_for_cropping = None
        self.show_scan_result_button.setVisible(False)
        # ### RESTORED: Hide labeling and cropping groups when cleared ###
        self.labeling_group.setVisible(False)
        self.cropping_group.setVisible(False)
        self.image_label.clear_selection()
        self._on_reset_crop() 
        
        # ### RESTORED: Clear text from label input widgets ###
        for input_widget in self.label_input_widgets.values():
            input_widget.clear()
        
        self._update_image_display()
        self.log_message("Cleared scan result, labels, and crop selection.")
    
    # ### RESTORED: Method to open the label management dialog ###
    def _on_manage_labels(self):
        dialog = ManageLabelsDialog(self.label_fields, self)
        dialog.label_fields_updated.connect(self._on_label_fields_updated)
        dialog.exec()

    # ### RESTORED: Slot to handle updates from the management dialog ###
    @pyqtSlot(list)
    def _on_label_fields_updated(self, new_fields: list):
        self.log_message("Label fields updated by user.")
        self.label_fields = new_fields
        self._update_labeling_panel()
        self._on_save_config()
        
    @pyqtSlot(bool)
    def _on_crop_mode_changed(self, is_box_mode: bool):
        """Switches between Bounding Box and Slider cropping modes."""
        if is_box_mode:
            self.crop_mode = "Bounding Box"
            self.image_label.set_drawing_enabled(True)
            self.sliders_panel.setEnabled(False)
        else:
            self.crop_mode = "Sliders"
            self.image_label.set_drawing_enabled(False)
            self.sliders_panel.setEnabled(True)
        self._trigger_redraw()

    @pyqtSlot(QRect)
    def _on_crop_box_drawn(self, rect_widget: QRect):
        """Handles the crop_rect_changed signal from the InteractiveCropLabel."""
        pixmap = self.image_label.pixmap()
        if not pixmap or pixmap.isNull(): return

        widget_w, widget_h = self.image_label.width(), self.image_label.height()
        pixmap_w, pixmap_h = pixmap.width(), pixmap.height()
        
        offset_x = (widget_w - pixmap_w) / 2
        offset_y = (widget_h - pixmap_h) / 2
        
        norm_x = (rect_widget.x() - offset_x) / pixmap_w
        norm_y = (rect_widget.y() - offset_y) / pixmap_h
        norm_w = rect_widget.width() / pixmap_w
        norm_h = rect_widget.height() / pixmap_h
        
        self.active_crop_rect_normalized = QRectF(norm_x, norm_y, norm_w, norm_h).normalized()
        
        self._sync_crop_ui_from_state() 
        self._trigger_redraw()

    @pyqtSlot()
    def _on_crop_slider_changed(self):
        """Updates the normalized crop rect when any slider is moved."""
        l = self.crop_slider_left.value() / 1000.0
        r = self.crop_slider_right.value() / 1000.0
        t = self.crop_slider_top.value() / 1000.0
        b = self.crop_slider_bottom.value() / 1000.0

        if l >= r:
            self.crop_slider_left.setValue(int(r * 1000) - 1)
            return
        if t >= b:
            self.crop_slider_top.setValue(int(b * 1000) - 1)
            return

        self.active_crop_rect_normalized = QRectF(l, t, r - l, b - t)
        
        self.crop_label_left.setText(f"{l:.1%}")
        self.crop_label_right.setText(f"{r:.1%}")
        self.crop_label_top.setText(f"{t:.1%}")
        self.crop_label_bottom.setText(f"{b:.1%}")
        
        self._trigger_redraw()

    def _sync_crop_ui_from_state(self):
        """Updates all crop UI elements from the active_crop_rect_normalized."""
        rect = self.active_crop_rect_normalized
        
        for slider in [self.crop_slider_left, self.crop_slider_right, self.crop_slider_top, self.crop_slider_bottom]:
            slider.blockSignals(True)
            
        self.crop_slider_left.setValue(int(rect.left() * 1000))
        self.crop_slider_right.setValue(int(rect.right() * 1000))
        self.crop_slider_top.setValue(int(rect.top() * 1000))
        self.crop_slider_bottom.setValue(int(rect.bottom() * 1000))

        self.crop_label_left.setText(f"{rect.left():.1%}")
        self.crop_label_right.setText(f"{rect.right():.1%}")
        self.crop_label_top.setText(f"{rect.top():.1%}")
        self.crop_label_bottom.setText(f"{rect.bottom():.1%}")

        if self.crop_mode == "Bounding Box":
            self.crop_mode_box_radio.setChecked(True)
        else:
            self.crop_mode_slider_radio.setChecked(True)

        for slider in [self.crop_slider_left, self.crop_slider_right, self.crop_slider_top, self.crop_slider_bottom]:
            slider.blockSignals(False)

    @pyqtSlot()
    def _on_reset_crop(self):
        """Resets the crop area to the full image."""
        self.active_crop_rect_normalized = QRectF(0.0, 0.0, 1.0, 1.0)
        self._sync_crop_ui_from_state()
        self._trigger_redraw()

    @pyqtSlot(bool)
    def _on_remember_crop_toggled(self, checked: bool):
        self.remember_crop_settings = checked
        self.log_message(f"Remember crop settings: {'Enabled' if checked else 'Disabled'}")
    
    @pyqtSlot(int, int, str)
    def _update_save_progress(self, current_step: int, total_steps: int, message: str):
        self.save_status_label.setText(message)
        if total_steps > 0:
            self.save_progress_bar.setRange(0, total_steps)
            self.save_progress_bar.setValue(current_step)

    # ### RESTORED: Method to save labels to the HDF5 file ###
    def _on_save_labels(self):
        """
        Efficiently updates the labels in the HDF5 file without rewriting the
        entire dataset.
        """
        if not self.last_h5_path or not os.path.exists(self.last_h5_path):
            QMessageBox.critical(self, "Error", "Cannot find the HDF5 file to save labels to. Please perform a scan or load a file first.")
            return

        try:
            labels_to_save = {}
            for key, input_widget in self.label_input_widgets.items():
                labels_to_save[key] = input_widget.text()
            
            labels_to_save["label_modification_date"] = datetime.now().isoformat()
            
            if not any(labels_to_save.values()):
                QMessageBox.warning(self, "Input Required", "Please fill in at least one label field before saving.")
                return

            self.log_message(f"Efficiently updating labels for {os.path.basename(self.last_h5_path)}...")
            
            # Call the new, efficient update function
            file_io.update_h5_labels(self.last_h5_path, labels_to_save)
            
            self.log_message("Labels successfully updated in the HDF5 file.")
            QMessageBox.information(self, "Success", "Labels were successfully saved to the HDF5 file.")

        except Exception as e:
            error_msg = f"An error occurred while saving labels:\n\n{e}"
            self.log_message(f"ERROR saving labels: {error_msg}")
            QMessageBox.critical(self, "Save Error", error_msg)

    def _on_save_config(self):
        config = self._get_current_config()
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, 'w') as f: json.dump(config, f, indent=4)
            self.log_message(f"Successfully saved configuration to {CONFIG_FILE}")
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Failed to save configuration file.\n{e}")
    
    def _get_default_calib_dir(self):
        calib_dir = ASSETS_DIR
        return calib_dir if os.path.isdir(calib_dir) else os.getcwd()

    def _load_default_calibration(self):
        default_path = os.path.join(self._get_default_calib_dir(), os.path.basename(MASTER_CALIBRATION_FILE))
        if os.path.exists(default_path):
            self.log_message(f"Found default {os.path.basename(MASTER_CALIBRATION_FILE)}, attempting to load...")
            self._load_and_validate_calibration_file(default_path)
        else:
            self.log_message(f"Default {os.path.basename(MASTER_CALIBRATION_FILE)} not found in {self._get_default_calib_dir()}")
            
    def _on_load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Master Calibration File", self._get_default_calib_dir(), "JSON Files (*.json)")
        if path:
            self._load_and_validate_calibration_file(path)
            
    @pyqtSlot(str)
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text_edit.append(f"[{timestamp}] {message}")

    def _on_connect_all(self):
        self.log_message("Attempting to connect all devices...")
        if self.camera.connect():
            min_exp, max_exp = self.camera.get_exposure_limits()
            if min_exp is not None and max_exp is not None:
                self.exposure_spinbox.setRange(min_exp, max_exp)
            self.camera.set_exposure_time(self.exposure_spinbox.value())
        
        self.stage.connect()
        self._update_ui_state()

    def _on_disconnect_all(self):
        self.log_message("Disconnecting all devices...");
        self.camera.disconnect()
        self.stage.disconnect()
        self._update_ui_state()

    @pyqtSlot(float)
    def _on_exposure_time_updated(self, actual_exposure_us: float):
        self.exposure_spinbox.blockSignals(True)
        self.exposure_spinbox.setValue(actual_exposure_us)
        self.exposure_spinbox.blockSignals(False)

    def _update_fps(self):
        elapsed_time = time.time() - self.fps_start_time
        if elapsed_time > 0: self.current_fps = self.frame_count / elapsed_time
        self.fps_status_label.setText(f"FPS: {self.current_fps:.2f}")
        self.frame_count = 0
        self.fps_start_time = time.time()

    def _on_start_live_view(self):
        self._on_clear_scan_result()
        self.log_message("Starting Live View...")
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_timer.start(1000)
        self.camera.start_live_view()
        self._update_ui_state()

    def _on_stop_live_view(self):
        self.log_message("Stopping Live View...")
        self.fps_timer.stop()
        self.current_fps = 0.0
        self.fps_status_label.setText("FPS: N/A")
        self.camera.stop_live_view()
        self._deactivate_spectral_overlays()
        self._update_ui_state()

    @pyqtSlot()
    def _on_homing_complete(self):
        self.current_stage_position = 0
        self.stage_current_pos_label.setText(f"{self.current_stage_position} steps")
        self.log_message("Stage homed. Current position set to 0.")
        self._update_ui_state()

    def _on_move_stage(self):
        position = self.stage_pos_spinbox.value()
        speed = self.stage_speed_spinbox.value()
        self.log_message(f"Sending command: Move to {position} at speed {speed}")
        self.stage.move_to(position, speed)
        self.current_stage_position = position
        self.stage_current_pos_label.setText(f"{self.current_stage_position} steps")
        self._update_ui_state()

    def _on_stage_limits_changed(self):
        min_limit = self.stage_min_limit_spinbox.value()
        max_limit = self.stage_max_limit_spinbox.value()
        if min_limit > max_limit:
            self.stage_min_limit_spinbox.setValue(max_limit)
            min_limit = max_limit
        self.stage_pos_spinbox.setRange(min_limit, max_limit)
        self.log_message(f"Manual stage movement limits updated to: [{min_limit}, {max_limit}]")

    def _on_set_start_pos(self):
        if self.current_stage_position is not None:
            self.scan_start_pos = self.current_stage_position
            self.scan_start_label.setText(f"{self.scan_start_pos} steps")
            self.log_message(f"Scan start position set to: {self.scan_start_pos}")
            self._update_ui_state()

    def _on_set_end_pos(self):
        if self.current_stage_position is not None:
            self.scan_end_pos = self.current_stage_position
            self.scan_end_label.setText(f"{self.scan_end_pos} steps")
            self.log_message(f"Scan end position set to: {self.scan_end_pos}")
            self._update_ui_state()

    def _on_acquire_data_cube(self):
        if self.camera.is_acquiring:
            self.log_message("Stopping live view to start acquisition...")
            self._on_stop_live_view()
            QTimer.singleShot(200, self._on_acquire_data_cube)
            return
        self.log_message("Launching Data Cube Acquisition dialog...")
        
        config = self._get_current_config()
        config['calibration_data'] = self.calibration_data
        
        # ### MODIFIED: Pass the label field structure to the dialog ###
        acquisition_dialog = AcquisitionDialog(self, self.camera, self.stage, config, self.label_fields)
        
        acquisition_dialog.acquisition_complete.connect(self._on_acquisition_complete)
        acquisition_dialog.data_ready_for_saving.connect(self._on_data_ready_for_saving)
        acquisition_dialog.scan_and_thread_finished.connect(self._on_scan_dialog_finished)
        acquisition_dialog.settings_save_requested.connect(self._on_save_acquisition_settings)
        # ### NEW: Connect signal to update label structure ###
        acquisition_dialog.label_fields_save_requested.connect(self._on_label_fields_updated_from_dialog)

        acquisition_dialog.exec()
        self.data_cube_save_dir = acquisition_dialog.dir_edit.text()
        self._update_ui_state()
    
    @pyqtSlot(list)
    def _on_label_fields_updated_from_dialog(self, new_fields: list):
        self.log_message("Label field structure updated from acquisition dialog.")
        self.label_fields = new_fields
        self._update_labeling_panel() # Update the main window's panel too
        self._on_save_config()

    @pyqtSlot(dict)
    def _on_save_acquisition_settings(self, settings: dict):
        self.log_message("Updating main config with new acquisition settings.")
        
        self.output_size_percentage = settings.get("output_size_percentage", self.output_size_percentage)
        self.compression_algo = settings.get("compression_algo", self.compression_algo)
        self.compression_level = settings.get("compression_level", self.compression_level)
        self.shuffle_enabled = settings.get("shuffle_enabled", self.shuffle_enabled)
        self.fletcher32_enabled = settings.get("fletcher32_enabled", self.fletcher32_enabled)
        self.batching_enabled = settings.get("batching_enabled", self.batching_enabled)
        self.batch_size = settings.get("batch_size", self.batch_size)
        self.data_cube_save_dir = settings.get("data_cube_save_dir", self.data_cube_save_dir)
        
        self._on_save_config()
        QMessageBox.information(self, "Settings Saved", "Acquisition and save settings have been saved to the configuration file.")

    @pyqtSlot(dict)
    def _on_scan_dialog_finished(self, config: dict):
        if config.get("return_to_start_after_scan", False):
            wait_s = config.get("post_scan_wait_seconds", 5)
            self.log_message(f"Scan finished. Staging return trip in {wait_s} seconds...")
            QTimer.singleShot(wait_s * 1000, lambda: self._execute_return_to_start(config))

    def _execute_return_to_start(self, config: dict):
        """Sends the move command for the return trip."""
        start_pos = config.get('start_position')
        return_speed = config.get('scan_speed', 200) 
        if start_pos is not None:
            self.log_message(f"Executing return to start position: {start_pos}")
            self.stage.move_to(start_pos, return_speed)
        else:
            self.log_message("Cannot return to start; start position not defined in config.")
    
    @pyqtSlot(dict)
    def _on_data_ready_for_saving(self, data_package: dict):
        self.log_message(f"Data package received. Starting background save to {os.path.basename(data_package['filepath'])}...")
        
        self.save_progress_group.setVisible(True)
        # ### RESTORED: Disable the save labels button during save ###
        self.save_labels_button.setEnabled(False)
        self.crop_save_button.setEnabled(False)

        self.save_thread = HDF5SaveThread(data_package, self)
        
        self.save_thread.progress_update.connect(self._update_save_progress)
        
        self.save_thread.finished.connect(self._on_save_finished)
        self.save_thread.error.connect(self._on_save_error)
        self.save_thread.finished.connect(self.save_thread.deleteLater)
        self.save_thread.start()

    @pyqtSlot(str)
    def _on_save_finished(self, message: str):
        self.log_message(f"✅ Background save successful: {message}")
        
        self.save_progress_group.setVisible(False)
        self.save_labels_button.setEnabled(True)
        self.crop_save_button.setEnabled(True)

    @pyqtSlot(str)
    def _on_save_error(self, error_message: str):
        self.log_message(f"❌ BACKGROUND SAVE FAILED: {error_message}")
        QMessageBox.critical(self, "Background Save Failed", error_message)
        
        self.save_progress_group.setVisible(False)
        self.save_labels_button.setEnabled(True)
        self.crop_save_button.setEnabled(True)

    @staticmethod
    def apply_geometric_corrections(frame: np.ndarray, calib_data: dict) -> np.ndarray:
        if not calib_data or "calibration_steps" not in calib_data: return frame
        try:
            calib_steps = calib_data["calibration_steps"]
            straighten_data = calib_steps.get("straightening")
            crop_data = calib_steps.get("cropping")
            straightened_frame = transformers.apply_straightening_to_image_array(frame, straighten_data)
            return transformers.apply_cropping(straightened_frame, crop_data)
        except Exception: return frame

    def _trigger_redraw(self):
        if self.last_frame is not None or self.last_scan_result is not None:
            self._update_image_display()

    @pyqtSlot(np.ndarray)
    def _update_image_display(self, new_frame: np.ndarray = None):
        if self.last_scan_result is not None:
            display_source = self.last_scan_result
        elif new_frame is not None:
            self.last_frame = new_frame.copy()
            display_source = self.last_frame
            self.frame_count += 1
        elif self.last_frame is not None:
            display_source = self.last_frame
        else:
            self.image_label.setText("No image data available.")
            self.image_label.setPixmap(QPixmap())
            return
            
        try:
            if display_source is self.last_frame and self.apply_geo_correction_checkbox.isChecked():
                display_frame = self.apply_geometric_corrections(display_source, self.calibration_data)
            else:
                display_frame = display_source

            if display_frame.ndim == 2:
                display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            else:
                display_frame_bgr = display_frame.copy()
            
            h, w, ch = display_frame_bgr.shape
            q_image = QImage(display_frame_bgr.data, w, h, ch * w, QImage.Format.Format_BGR888)
            base_pixmap = QPixmap.fromImage(q_image)

            RULER_WIDTH = 60
            has_ruler = self.wavelength_ruler_checkbox.isChecked() and self.calibration_data
            canvas_w = w + RULER_WIDTH if has_ruler else w
            canvas_h = h
            canvas_pixmap = QPixmap(canvas_w, canvas_h)
            canvas_pixmap.fill(QColor(45, 45, 45))

            painter = QPainter(canvas_pixmap)
            img_start_x = RULER_WIDTH if has_ruler else 0
            painter.drawPixmap(img_start_x, 0, base_pixmap)

            if has_ruler:
                self._draw_wavelength_ruler(painter, canvas_h, display_frame.shape[0])
            
            self._draw_view_overlays(painter, img_start_x, w, h, display_frame.shape[0])

            if self.last_scan_result is not None:
                self._draw_crop_preview(painter, img_start_x, w, h)

            painter.end()

            self.image_label.setPixmap(canvas_pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            if self.image_label.pixmap(): self.img_size_status_label.setText(f"Display Size: {self.image_label.pixmap().width()}x{self.image_label.pixmap().height()}")

        except Exception as e:
            self.log_message(f"ERROR during image display update: {e}\n{traceback.format_exc()}")
            self.image_label.setText("Error rendering frame.")

    def _draw_wavelength_ruler(self, painter: QPainter, canvas_h: int, display_h: int):
        calib_steps = self.calibration_data["calibration_steps"]
        calib_output_h = calib_steps["output_dimensions"]["height"]
        coeffs = calib_steps.get("spectral", {}).get("coefficients")
        if not (coeffs and calib_output_h > 1 and display_h > 1): return
        
        painter.setPen(QColor(220, 220, 220))
        painter.setFont(self.font())

        for target_wav in np.arange(400, 1001, 50):
            pixel_y_calib = DataCubeAcquisitionThread.get_pixel_for_wavelength(target_wav, coeffs, calib_output_h)
            if pixel_y_calib is not None:
                display_y = int(round(pixel_y_calib * (display_h - 1) / (calib_output_h - 1)))
                painter.drawLine(50, display_y, 60, display_y)
                painter.drawText(5, display_y + 4, str(int(target_wav)))

    def _draw_view_overlays(self, painter: QPainter, start_x: int, width: int, height: int, display_h: int):
        if self.grid_checkbox.isChecked():
            painter.setPen(QPen(QColor(0, 100, 0, 128), 1))
            for i in range(1, 10):
                x_pos, y_pos = start_x + (width * i // 10), height * i // 10
                painter.drawLine(x_pos, 0, x_pos, height)
                painter.drawLine(start_x, y_pos, start_x + width, y_pos)
        
        if self.center_line_checkbox.isChecked():
            painter.setPen(QPen(QColor(255, 0, 0, 180), 1))
            painter.drawLine(start_x + (width // 2), 0, start_x + (width // 2), height)
            
        if self.spectral_lines_checkbox.isChecked() and self.calibration_data:
            calib_steps = self.calibration_data["calibration_steps"]
            coeffs = calib_steps.get("spectral", {}).get("coefficients")
            calib_output_h = calib_steps.get("output_dimensions", {}).get("height")
            if not (coeffs and calib_output_h and calib_output_h > 1 and display_h > 1): return
            
            scale_factor = (display_h - 1) / (calib_output_h - 1)
            for wav, color in [(self.r_wavelength_spinbox.value(), QColor(255, 0, 0)),
                               (self.g_wavelength_spinbox.value(), QColor(0, 255, 0)),
                               (self.b_wavelength_spinbox.value(), QColor(0, 0, 255))]:
                pix_calib = DataCubeAcquisitionThread.get_pixel_for_wavelength(wav, coeffs, calib_output_h)
                if pix_calib:
                    pix_display = int(round(pix_calib * scale_factor))
                    painter.setPen(QPen(color, 1))
                    painter.drawLine(start_x, pix_display, start_x + width, pix_display)
    
    def _draw_crop_preview(self, painter: QPainter, start_x: int, width: int, height: int):
        """Draws a semi-transparent overlay to highlight the crop area."""
        norm_rect = self.active_crop_rect_normalized
        if norm_rect.width() >= 1.0 and norm_rect.height() >= 1.0:
            return

        crop_x = start_x + int(norm_rect.x() * width)
        crop_y = int(norm_rect.y() * height)
        crop_w = int(norm_rect.width() * width)
        crop_h = int(norm_rect.height() * height)
        
        crop_area = QRect(crop_x, crop_y, crop_w, crop_h)
        full_area = QRect(start_x, 0, width, height)
        
        path = QPainterPath()
        path.addRect(QRectF(full_area))
        path.addRect(QRectF(crop_area))
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 120)) 
        painter.drawPath(path)
        
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(50, 200, 255), 2, Qt.PenStyle.SolidLine))
        painter.drawRect(crop_area)

    def _on_camera_connection_lost(self, message: str):
        QMessageBox.critical(self, "Camera Connection Lost", message)
        self.camera.disconnect(); self._update_ui_state()

    def _on_stage_connection_lost(self):
        QMessageBox.critical(self, "Stage Connection Lost", "Connection to the Arduino stage was lost.")
        self.stage.disconnect(); self._update_ui_state()

    def closeEvent(self, event):
        self.log_message("Application closing, disconnecting all devices...")
        self._on_disconnect_all()
        event.accept()

    # ### RESTORED: Method to dynamically create the label input panel ###
    def _update_labeling_panel(self):
        # Clear existing widgets from the layout
        while self.label_widgets_layout.count():
            item = self.label_widgets_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.label_input_widgets.clear()
        
        # Create and add new widgets based on the current self.label_fields
        for field in self.label_fields:
            key = field.get("key")
            if not key: continue
            
            label_widget = QLabel(f"{key}:")
            input_widget = QLineEdit()
            input_widget.setPlaceholderText(field.get("description", ""))
            
            self.label_widgets_layout.addRow(label_widget, input_widget)
            self.label_input_widgets[key] = input_widget
    
    def _deactivate_spectral_overlays(self):
        """Turns off spectral-only overlay controls. To be called when leaving live view."""
        self.wavelength_ruler_checkbox.setChecked(False)
        self.spectral_lines_checkbox.setChecked(False)
    
    @staticmethod
    def _create_rgb_strip_from_frame(frame: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
        """Helper to create one RGB strip from a single processed HSI frame."""
        current_spectral_height = frame.shape[0]
        calib_output_h = params['calib_output_height']
        
        r_pix_calib = DataCubeAcquisitionThread.get_pixel_for_wavelength(params['r_wavelength'], params['spectral_calib']['coefficients'], calib_output_h)
        g_pix_calib = DataCubeAcquisitionThread.get_pixel_for_wavelength(params['g_wavelength'], params['spectral_calib']['coefficients'], calib_output_h)
        b_pix_calib = DataCubeAcquisitionThread.get_pixel_for_wavelength(params['b_wavelength'], params['spectral_calib']['coefficients'], calib_output_h)
        
        if any(p is None for p in [r_pix_calib, g_pix_calib, b_pix_calib]):
            raise ValueError("Could not map one or more RGB wavelengths to pixel rows.")
        
        if calib_output_h > 1 and current_spectral_height > 1:
            scale_factor = (current_spectral_height - 1) / (calib_output_h - 1)
            r_pix = int(round(r_pix_calib * scale_factor))
            g_pix = int(round(g_pix_calib * scale_factor))
            b_pix = int(round(b_pix_calib * scale_factor))
        else:
            r_pix, g_pix, b_pix = r_pix_calib, g_pix_calib, b_pix_calib

        half_thick = params['thickness'] // 2
        r_strip = np.mean(frame[max(0, r_pix - half_thick):min(current_spectral_height, r_pix + half_thick + 1), :], axis=0)
        g_strip = np.mean(frame[max(0, g_pix - half_thick):min(current_spectral_height, g_pix + half_thick + 1), :], axis=0)
        b_strip = np.mean(frame[max(0, b_pix - half_thick):min(current_spectral_height, b_pix + half_thick + 1), :], axis=0)
        
        single_row_image = np.stack([b_strip, g_strip, r_strip], axis=-1).reshape(1, -1, 3).astype(np.uint8)
        
        sampling_info = {
            'r_pixel_row': r_pix, 'g_pixel_row': g_pix, 'b_pixel_row': b_pix,
            'info': 'Pixel row indices used for sampling the RGB preview from the data cube.'
        }
        return single_row_image, sampling_info

    @pyqtSlot()
    def _on_crop_and_save(self):
        """Core logic to perform the crop and save a new HDF5 file."""
        if not self.last_h5_path or self.active_crop_rect_normalized.isNull():
            self.log_message("Crop Error: No source file or valid crop area defined.")
            return

        if self.data_cube_for_cropping is None:
            try:
                self.log_message(f"Loading full data cube from {os.path.basename(self.last_h5_path)} for cropping...")
                self.data_cube_for_cropping, _, _, _ = file_io.load_h5(self.last_h5_path)
            except Exception as e:
                self.log_message(f"FATAL: Could not load data cube for cropping. {e}")
                QMessageBox.critical(self, "Load Error", f"Could not load data cube from file for cropping:\n{e}")
                return
        
        try:
            full_spectral, full_spatial, full_bands = self.data_cube_for_cropping.shape
            norm_rect = self.active_crop_rect_normalized
            
            spatial_start = int(np.floor(norm_rect.left() * full_spatial))
            spatial_end = int(np.ceil(norm_rect.right() * full_spatial))
            band_start = int(np.floor(norm_rect.top() * full_bands))
            band_end = int(np.ceil(norm_rect.bottom() * full_bands))

            spatial_start, spatial_end = max(0, spatial_start), min(full_spatial, spatial_end)
            band_start, band_end = max(0, band_start), min(full_bands, band_end)
            
            if spatial_start >= spatial_end or band_start >= band_end:
                raise ValueError("Crop area resulted in zero size.")

            self.log_message(f"Cropping data cube: Spatial [{spatial_start}:{spatial_end}], Bands [{band_start}:{band_end}]")
            cropped_cube = self.data_cube_for_cropping[:, spatial_start:spatial_end, band_start:band_end]
            cropped_spectral, cropped_spatial, cropped_bands = cropped_cube.shape

            original_scan_params = self.last_scan_metadata.get("scan_parameters", {})
            original_fov_width_mm = original_scan_params.get('camera_fov_width_mm', 10.0)
            original_scan_dist_mm = abs(original_scan_params.get('end_position', 0) - original_scan_params.get('start_position', 0)) * original_scan_params.get('mm_per_step', 0.01)
            
            new_fov_width_mm = original_fov_width_mm * (cropped_spatial / full_spatial)
            new_scan_dist_mm = original_scan_dist_mm * (cropped_bands / full_bands)
            
            aspect_ratio_params = {
                'fov_width_mm': new_fov_width_mm,
                'scan_dist_mm': new_scan_dist_mm
            }
            
            self.log_message("Regenerating RGB preview from cropped data cube...")
            cropped_rgb_preview = self._create_rgb_from_cube(cropped_cube, aspect_ratio_params)
            
            original_path = self.last_h5_path
            new_path, _ = QFileDialog.getSaveFileName(
                self, "Save Cropped Data Cube As...", 
                os.path.join(os.path.dirname(original_path), f"{os.path.splitext(os.path.basename(original_path))[0]}_cropped.h5"),
                "HDF5 Files (*.h5)"
            )
            if not new_path: return

            _, original_metadata, original_labels, _ = file_io.load_h5(original_path)
            new_metadata = original_metadata.copy()
            new_metadata['shape'] = cropped_cube.shape
            new_metadata['description'] = f"Cropped from original file: {os.path.basename(original_path)}"
            new_metadata['cropping_info'] = {
                'original_shape': (full_spectral, full_spatial, full_bands),
                'normalized_rect': [norm_rect.x(), norm_rect.y(), norm_rect.width(), norm_rect.height()],
                'pixel_bounds': {'spatial': [spatial_start, spatial_end], 'bands': [band_start, band_end]}
            }

            new_data_package = {
                "filepath": new_path, "data_cube": cropped_cube, "metadata": new_metadata,
                "labels": original_labels, "rgb_preview": cropped_rgb_preview
            }
            self._on_data_ready_for_saving(new_data_package)

        except Exception as e:
            self.log_message(f"ERROR during cropping: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Cropping Error", f"An error occurred during the cropping process:\n{e}")

    def _on_launch_slice_viewer(self):
        """Opens a file dialog and launches the DataCubeSliceViewerDialog."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data Cube for Slice Analysis",
            self.data_cube_save_dir,
            "HDF5 Files (*.h5 *.hdf5)"
        )
        if filepath:
            try:
                viewer = DataCubeSliceViewerDialog(filepath, self)
                viewer.exec()
            except Exception as e:
                self.log_message(f"ERROR: Failed to launch slice viewer: {e}")
                QMessageBox.critical(self, "Viewer Error", f"Could not open the slice viewer:\n\n{e}")

    def _create_rgb_from_cube(self, data_cube: np.ndarray, aspect_ratio_params: dict = None) -> np.ndarray:
        """Regenerates a full RGB preview image from a data cube in memory."""
        if self.last_scan_metadata is None:
            raise ValueError("Cannot regenerate RGB preview without original scan metadata.")
        
        scan_params = self.last_scan_metadata.get('scan_parameters', {})
        rgb_sampling_config = {
            'r_wavelength': scan_params.get('r_wavelength', 640.0),
            'g_wavelength': scan_params.get('g_wavelength', 550.0),
            'b_wavelength': scan_params.get('b_wavelength', 460.0),
            'thickness': scan_params.get('sampling_thickness', 5),
            'spectral_calib': self.calibration_data['calibration_steps']['spectral'],
            'calib_output_height': self.calibration_data['calibration_steps']['output_dimensions']['height']
        }
        
        cube_transposed = data_cube.transpose(2, 0, 1)
        collected_strips = [self._create_rgb_strip_from_frame(frame, rgb_sampling_config)[0] for frame in cube_transposed]
        
        if not collected_strips:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        rgb_raw = np.vstack(collected_strips)
        
        if aspect_ratio_params:
            fov_width_mm = aspect_ratio_params['fov_width_mm']
            scan_dist_mm = aspect_ratio_params['scan_dist_mm']
        else:
            start_pos = scan_params.get('start_position', 0)
            end_pos = scan_params.get('end_position', 0)
            scan_dist_mm = abs(end_pos - start_pos) * scan_params.get('mm_per_step', 0.01)
            fov_width_mm = scan_params.get('camera_fov_width_mm', 10.0)
        
        if scan_dist_mm > 0 and fov_width_mm > 0:
            aspect_ratio = fov_width_mm / scan_dist_mm
            h, _ = rgb_raw.shape[:2]
            target_w = int(h * aspect_ratio)
            rgb_resized = cv2.resize(rgb_raw, (target_w, h), interpolation=cv2.INTER_AREA)
        else:
            rgb_resized = rgb_raw
            
        return cv2.normalize(rgb_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HsiControlApp()
    window.show()
    sys.exit(app.exec())