
"""
Interactive PyQt6 Test Application for CameraController.

This application provides a graphical user interface to test the functionality
of the CameraController class in real-time. It features a scalable, tabbed
interface and advanced UX controls:
- A live spectral profile plot, averaging the spatial axis and mapping to wavelength.
- A highly organized and maintainable UI codebase.
- A default path for loading the master calibration file.
- All original features (connection, live view, aggregation, saving, etc.).
"""
import sys
import os
from datetime import datetime
import time
from collections import deque
import numpy as np
import cv2
import json

# Ensure the script's directory is in the path to find local modules
# This makes imports like 'transformers' and 'hardware' more robust.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QFormLayout,
    QSpinBox, QFileDialog, QMessageBox, QDoubleSpinBox, QCheckBox,
    QStatusBar, QComboBox, QTabWidget
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

# Make sure the 'hardware' and local directories are accessible
from hardware.camera_controller import CameraController
from core import transformers

class SpectralProfileWidget(QWidget):
    """A widget that displays a wavelength-calibrated spectral profile."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 120)
        self.setMaximumHeight(150)
        self.profile = None
        self.wavelength_axis = None

    def update_profile(self, image: np.ndarray, calib_data: dict | None):
        """Calculates and redraws the spectral profile for the given image."""
        if image is None or image.size == 0 or calib_data is None or \
           "spectral" not in calib_data.get("calibration_steps", {}):
            self.profile = None
            self.wavelength_axis = None
            self.update()
            return
        
        profile = np.mean(image, axis=1)
        cv2.normalize(profile, profile, 0, self.height() - 20, cv2.NORM_MINMAX)
        self.profile = profile.astype(int)

        try:
            pixel_rows = np.arange(image.shape[0])
            spectral_cal = calib_data["calibration_steps"]["spectral"]
            self.wavelength_axis = transformers.map_pixel_to_wavelength(pixel_rows, spectral_cal)
        except (KeyError, ValueError):
            self.profile = None
            self.wavelength_axis = None
        
        self.update()

    def paintEvent(self, event):
        """Renders the spectral profile plot."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("black"))

        if self.profile is None or self.wavelength_axis is None:
            painter.setPen(QColor("gray"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load Spectral Calibration")
            return

        pen = QPen(QColor("#00aaff"), 1.5)
        painter.setPen(pen)
        
        points = [
            QPointF(i, self.height() - p - 10) for i, p in enumerate(self.profile)
        ]
        scale_x = self.width() / len(points) if len(points) > 1 else 1
        for p in points:
            p.setX(p.x() * scale_x)

        painter.drawPolyline(points)

        painter.setPen(QColor("white"))
        min_wl = self.wavelength_axis[0]
        max_wl = self.wavelength_axis[-1]
        painter.drawText(5, self.height() - 2, f"{min_wl:.0f} nm")
        painter.drawText(self.width() - 50, self.height() - 2, f"{max_wl:.0f} nm")


class InteractiveImageLabel(QLabel):
    """A QLabel that tracks mouse movements and emits pixel coordinates."""
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #333; color: white;")
        self.setMinimumSize(640, 480)

    def mouseMoveEvent(self, event):
        self.mouse_moved.emit(event.pos().x(), event.pos().y())
        super().mouseMoveEvent(event)
    
    def leaveEvent(self, event):
        self.mouse_moved.emit(-1, -1)
        super().leaveEvent(event)


class CameraTestApp(QMainWindow):
    """Main window for the Camera Controller test application."""

    def __init__(self, file_dir=os.path.dirname(os.path.abspath(__file__))):
        super().__init__()
        self.setWindowTitle("Dedicated Camera App")
        self.setGeometry(100, 100, 1200, 800)
        self.controller = CameraController(self)
        self.file_dir = file_dir
        self.final_aggregated_image = None
        self.last_frame = None
        self.processed_frame = None
        self.calibration_data = None
        self.frame_times = deque(maxlen=30)
        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

    def _create_widget(self, widget_class, **properties):
        """Generic widget factory for cleaner code."""
        widget = widget_class()
        for key, value in properties.items():
            setter_name = f"set{key}"
            if hasattr(widget, setter_name):
                setter = getattr(widget, setter_name)
                if isinstance(value, (list, tuple)):
                    setter(*value)
                else:
                    setter(value)
            else:
                widget.setProperty(key, value)
        return widget

    def _setup_ui(self):
        """Create and arrange all the widgets in the window."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        controls_tabs = QTabWidget()
        controls_tabs.setMaximumWidth(350)

        # --- Tab 1: Camera ---
        cam_tab = QWidget()
        cam_layout = QVBoxLayout(cam_tab)
        
        conn_group = QGroupBox("Connection")
        conn_layout = QVBoxLayout(conn_group)
        self.connect_button = QPushButton("Connect to Camera")
        self.disconnect_button = QPushButton("Disconnect")
        conn_layout.addWidget(self.connect_button)
        conn_layout.addWidget(self.disconnect_button)

        settings_group = QGroupBox("Camera Settings")
        settings_layout = QFormLayout(settings_group)
        self.exposure_spinbox = self._create_widget(QDoubleSpinBox, Range=(1.0, 1000000.0), Value=8000, Suffix=" µs", SingleStep=100, Decimals=0)
        settings_layout.addRow("Exposure Time:", self.exposure_spinbox)

        live_view_group = QGroupBox("Live View")
        live_view_layout = QVBoxLayout(live_view_group)
        self.start_live_view_button = QPushButton("Start Live View")
        self.stop_live_view_button = QPushButton("Stop Live View")
        live_view_layout.addWidget(self.start_live_view_button)
        live_view_layout.addWidget(self.stop_live_view_button)
        live_view_layout.addWidget(QLabel("(Press 'S' to save a snapshot)", font=QFont("Arial", 8, italic=True)))

        cam_layout.addWidget(conn_group)
        cam_layout.addWidget(settings_group)
        cam_layout.addWidget(live_view_group)
        cam_layout.addStretch()

        # --- Tab 2: Display & Calibration ---
        display_tab = QWidget()
        display_layout = QVBoxLayout(display_tab)
        
        display_options_group = QGroupBox("Display Options")
        display_options_layout = QFormLayout(display_options_group)
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_lines_spinbox = self._create_widget(QSpinBox, Range=(2, 50), Value=10)
        display_options_layout.addRow(self.grid_checkbox)
        display_options_layout.addRow("Grid Lines:", self.grid_lines_spinbox)
        
        transform_group = QGroupBox("Calibration")
        transform_layout = QFormLayout(transform_group)
        self.load_calib_button = QPushButton("Load Master Calibration")
        self.apply_transform_checkbox = QCheckBox("Apply Calibration Pipeline")
        self.calib_info_label = QLabel("No file loaded.", styleSheet="font-style: italic; color: #888;")
        transform_layout.addRow(self.load_calib_button)
        transform_layout.addRow(self.apply_transform_checkbox)
        transform_layout.addRow(self.calib_info_label)

        profile_group = QGroupBox("Live Spectral Profile")
        profile_layout = QVBoxLayout(profile_group)
        self.spectral_widget = SpectralProfileWidget()
        profile_layout.addWidget(self.spectral_widget)

        display_layout.addWidget(display_options_group)
        display_layout.addWidget(transform_group)
        display_layout.addWidget(profile_group)
        display_layout.addStretch()

        # --- Tab 3: Processing & Saving ---
        proc_tab = QWidget()
        proc_layout = QVBoxLayout(proc_tab)
        
        agg_group = QGroupBox("Aggregation Controls")
        agg_layout = QFormLayout(agg_group)
        self.agg_threshold_spinbox = self._create_widget(QSpinBox, Range=(1, 254), Value=20)
        self.start_agg_button = QPushButton("Start Aggregation")
        self.stop_agg_button = QPushButton("Stop Aggregation")
        self.save_agg_button = QPushButton("Save Final Image")
        agg_layout.addRow("Threshold:", self.agg_threshold_spinbox)
        agg_layout.addRow(self.start_agg_button)
        agg_layout.addRow(self.stop_agg_button)
        agg_layout.addRow(self.save_agg_button)

        snapshot_group = QGroupBox("Snapshot Options")
        snapshot_layout = QFormLayout(snapshot_group)
        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["TIFF", "PNG", "JPEG"])
        self.save_size_combo = QComboBox()
        self.save_size_combo.addItems(["Original"])
        snapshot_layout.addRow("Save Format:", self.save_format_combo)
        snapshot_layout.addRow("Save Size:", self.save_size_combo)

        proc_layout.addWidget(agg_group)
        proc_layout.addWidget(snapshot_group)
        proc_layout.addStretch()

        # --- Assemble UI ---
        controls_tabs.addTab(cam_tab, "Camera")
        controls_tabs.addTab(display_tab, "Display")
        controls_tabs.addTab(proc_tab, "Processing")
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit(readOnly=True)
        log_layout.addWidget(self.log_text_edit)
        left_layout = QVBoxLayout()
        left_layout.addWidget(controls_tabs)
        left_layout.addWidget(log_group)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        display_layout = QVBoxLayout()
        self.image_label = InteractiveImageLabel("Connect camera and start a view")
        display_layout.addWidget(self.image_label)
        self.setStatusBar(QStatusBar(self))
        self.img_size_label = QLabel("Image Size: N/A")
        self.fps_label = QLabel("FPS: N/A")
        self.coords_label = QLabel("Coords: N/A")
        self.statusBar().addPermanentWidget(self.img_size_label, 1)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().addPermanentWidget(self.coords_label)
        main_layout.addWidget(left_widget)
        main_layout.addLayout(display_layout)

    def _connect_signals(self):
        self.controller.status_update.connect(self.log_message)
        self.controller.connection_lost.connect(self._on_connection_lost)
        self.controller.new_live_frame.connect(self._update_image_display)
        self.controller.aggregation_updated.connect(self._update_image_display)
        self.controller.aggregation_finished.connect(self._on_aggregation_finished)
        self.controller.exposure_time_updated.connect(self._on_exposure_time_updated)
        self.connect_button.clicked.connect(self._on_connect_clicked)
        self.disconnect_button.clicked.connect(self.controller.disconnect)
        self.start_live_view_button.clicked.connect(self._on_start_live_view)
        self.stop_live_view_button.clicked.connect(self._on_stop_live_view)
        self.start_agg_button.clicked.connect(self._on_start_aggregation)
        self.stop_agg_button.clicked.connect(self._on_stop_aggregation)
        self.save_agg_button.clicked.connect(self._save_aggregated_image)
        self.exposure_spinbox.valueChanged.connect(self._on_exposure_changed)
        self.grid_checkbox.stateChanged.connect(self._on_display_options_changed)
        self.grid_lines_spinbox.valueChanged.connect(self._on_display_options_changed)
        self.load_calib_button.clicked.connect(self._load_calibration)
        self.apply_transform_checkbox.stateChanged.connect(self._on_display_options_changed)
        self.image_label.mouse_moved.connect(self._on_mouse_moved_on_image)

    def _update_ui_state(self):
        is_connected = self.controller.is_connected
        is_acquiring = self.controller.is_acquiring
        is_aggregating = self.controller.is_aggregating
        is_idle = not is_acquiring and not is_aggregating
        has_calibration = self.calibration_data is not None
        self.connect_button.setEnabled(not is_connected)
        self.disconnect_button.setEnabled(is_connected)
        self.exposure_spinbox.setEnabled(is_connected)
        self.grid_checkbox.setEnabled(is_connected)
        self.grid_lines_spinbox.setEnabled(is_connected)
        self.start_live_view_button.setEnabled(is_connected and is_idle)
        self.stop_live_view_button.setEnabled(is_acquiring)
        self.start_agg_button.setEnabled(is_connected and is_idle)
        self.stop_agg_button.setEnabled(is_aggregating)
        self.save_agg_button.setEnabled(is_idle and self.final_aggregated_image is not None)
        self.load_calib_button.setEnabled(True)
        self.apply_transform_checkbox.setEnabled(is_connected and has_calibration)
        self.save_format_combo.setEnabled(is_connected)
        self.save_size_combo.setEnabled(is_connected)

    def _load_calibration(self):
        script_dir = self.file_dir
        #default_path = os.path.join(script_dir, '..', 'calibration_wizard', 'assets', 'master_calibration.json')
        default_path = os.path.join(script_dir, 'assets', 'master_calibration.json')
        if not os.path.exists(default_path):
            print("didn't exist")
            default_path = os.getcwd()

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Master Calibration File", default_path, "JSON Files (*.json)")
        if not file_path: return
        try:
            with open(file_path, 'r') as f: data = json.load(f)
            if "calibration_steps" not in data or "straightening" not in data["calibration_steps"] or "cropping" not in data["calibration_steps"]:
                raise ValueError("JSON is missing required 'straightening' or 'cropping' steps.")
            self.calibration_data = data
            self.log_message(f"Successfully loaded calibration from: {os.path.basename(file_path)}")
            self.calib_info_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.calib_info_label.setStyleSheet("font-style: italic; color: #333;")
            self.save_size_combo.blockSignals(True)
            if self.save_size_combo.count() > 1: self.save_size_combo.removeItem(1)
            crop_box = self.calibration_data["calibration_steps"]["cropping"]["bbox_pixels"]
            size_str = f"Calibrated ({crop_box['width']}x{crop_box['height']})"
            self.save_size_combo.addItem(size_str)
            self.save_size_combo.blockSignals(False)
        except Exception as e:
            self.calibration_data = None
            self.calib_info_label.setText("Load failed.")
            self.calib_info_label.setStyleSheet("font-style: italic; color: red;")
            error_msg = f"Failed to load or parse calibration file.\nError: {e}"
            self.log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Calibration Error", error_msg)
        finally:
            self._update_ui_state()
            self._on_display_options_changed()

    def _update_image_display(self, frame: np.ndarray):
        if frame is None: return
        self.last_frame = frame.copy()
        current_time = time.monotonic()
        self.frame_times.append(current_time)
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0: self.fps_label.setText(f"FPS: {(len(self.frame_times) - 1) / time_diff:.2f}")
        h, w = frame.shape[:2]
        self.img_size_label.setText(f"Image Size: {w}x{h}")
        if self.apply_transform_checkbox.isChecked() and self.calibration_data:
            try:
                straighten_data = self.calibration_data["calibration_steps"]["straightening"]
                straightened_img = transformers.apply_straightening_to_image_array(frame, straighten_data)
                crop_data = self.calibration_data["calibration_steps"]["cropping"]
                processed_frame = transformers.apply_cropping(straightened_img, crop_data)
            except Exception as e:
                self.log_message(f"ERROR applying transform: {e}")
                processed_frame = frame
        else:
            processed_frame = frame
        
        self.processed_frame = processed_frame.copy()
        
        # ### FIX ### This now correctly calls the spectral_widget and its update_profile method.
        self.spectral_widget.update_profile(self.processed_frame, self.calibration_data)
        
        display_frame = self.processed_frame
        if self.grid_checkbox.isChecked():
            if len(display_frame.shape) == 2: display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            h_disp, w_disp = display_frame.shape[:2]
            grid_color = (0, 255, 0)
            grid_thickness = 1
            num_lines = self.grid_lines_spinbox.value()
            for i in range(1, num_lines + 1):
                x = int(w_disp * i / (num_lines + 1))
                cv2.line(display_frame, (x, 0), (x, h_disp), grid_color, grid_thickness)
            for i in range(1, num_lines + 1):
                y = int(h_disp * i / (num_lines + 1))
                cv2.line(display_frame, (0, y), (w_disp, y), grid_color, grid_thickness)
        
        if len(display_frame.shape) == 3:
            h_disp, w_disp, ch = display_frame.shape
            q_image = QImage(display_frame.data, w_disp, h_disp, ch * w_disp, QImage.Format.Format_BGR888)
        else:
            h_disp, w_disp = display_frame.shape
            q_image = QImage(display_frame.data, w_disp, h_disp, w_disp, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def _on_mouse_moved_on_image(self, x_label, y_label):
        if x_label < 0 or self.processed_frame is None or self.image_label.pixmap() is None:
            self.coords_label.setText("Coords: N/A")
            return
        
        pixmap = self.image_label.pixmap()
        x_pix = int(x_label * pixmap.width() / self.image_label.width())
        y_pix = int(y_label * pixmap.height() / self.image_label.height())
        
        x_img = int(x_pix * self.processed_frame.shape[1] / pixmap.width())
        y_img = int(y_pix * self.processed_frame.shape[0] / pixmap.height())

        if 0 <= x_img < self.processed_frame.shape[1] and 0 <= y_img < self.processed_frame.shape[0]:
            info = f"Coords: ({x_img}, {y_img})"
            if self.calibration_data and "spectral" in self.calibration_data["calibration_steps"]:
                try:
                    spectral_data = self.calibration_data["calibration_steps"]["spectral"]
                    wavelength = transformers.map_pixel_to_wavelength(y_img, spectral_data)
                    info += f" | λ: {wavelength:.2f} nm"
                except (ValueError, KeyError):
                    pass
            self.coords_label.setText(info)
        else:
            self.coords_label.setText("Coords: N/A")

    def log_message(self, message: str):
        self.log_text_edit.append(message)
        if "connected" in message or "disconnected" in message:
            self._update_ui_state()
            
    def _on_connect_clicked(self):
        if self.controller.connect():
            self.log_message("Connection successful.")
            min_exp, max_exp = self.controller.get_exposure_limits()
            if min_exp is not None and max_exp is not None:
                self.log_message(f"Camera exposure range: {min_exp:.0f} to {max_exp:.0f} µs.")
                self.exposure_spinbox.setRange(min_exp, max_exp)
            self._on_exposure_changed(self.exposure_spinbox.value())
        else:
            self.log_message("Connection failed. Check log for details.")
        self._update_ui_state()

    def _on_exposure_time_updated(self, actual_exposure_us: float):
        self.exposure_spinbox.blockSignals(True)
        self.exposure_spinbox.setValue(actual_exposure_us)
        self.exposure_spinbox.blockSignals(False)

    def _on_start_live_view(self):
        self.log_message("--- Starting Live View ---")
        self.final_aggregated_image = None
        self.last_frame = None
        self.processed_frame = None
        self.frame_times.clear()
        self.controller.start_live_view()
        self._update_ui_state()

    def _on_stop_live_view(self):
        self.log_message("--- Stopping Live View ---")
        self.controller.stop_live_view()
        self.fps_label.setText("FPS: N/A")
        self._update_ui_state()

    def _on_start_aggregation(self):
        self.log_message("--- Starting Aggregation ---")
        self.final_aggregated_image = None
        self.last_frame = None
        self.processed_frame = None
        threshold = self.agg_threshold_spinbox.value()
        self.controller.start_aggregation(threshold)
        self._update_ui_state()

    def _on_stop_aggregation(self):
        self.log_message("--- Stopping Aggregation ---")
        self.controller.stop_aggregation()

    def _on_aggregation_finished(self, image_array: np.ndarray):
        self.log_message(">>> Aggregation finished. Final image is ready to be saved.")
        self.final_aggregated_image = image_array
        self._update_ui_state()

    def _on_connection_lost(self, message: str):
        QMessageBox.critical(self, "Connection Lost", message)
        self.controller.disconnect()
        self.last_frame = None
        self.processed_frame = None
        self._update_ui_state()

    def _on_exposure_changed(self, value: float):
        self.controller.set_exposure_time(value)

    def _on_display_options_changed(self):
        if self.last_frame is not None:
            self._update_image_display(self.last_frame)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            if self.controller.is_acquiring and self.processed_frame is not None:
                self._save_snapshot()
                event.accept()
            else:
                self.log_message("Snapshot key 'S' pressed, but live view is not active.")
                event.ignore()
        else:
            super().keyPressEvent(event)

    def _save_snapshot(self):
        output_dir = "snapshots"
        try:
            os.makedirs(output_dir, exist_ok=True)
            save_format = self.save_format_combo.currentText().lower()
            extension = "jpg" if save_format == "jpeg" else "tif" if save_format == "tiff" else "png"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            file_path = os.path.join(output_dir, f"snapshot_{timestamp}.{extension}")
            
            if self.save_size_combo.currentIndex() == 0:
                image_to_save = self.last_frame
            else:
                image_to_save = self.processed_frame
                
            if image_to_save is None:
                self.log_message("ERROR: No valid frame to save.")
                return

            cv2.imwrite(file_path, image_to_save)
            self.log_message(f"Snapshot saved to: {file_path}")
        except Exception as e:
            error_msg = f"Failed to save snapshot.\nError: {e}"
            self.log_message(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Snapshot Error", error_msg)

    def _save_aggregated_image(self):
        if self.final_aggregated_image is None:
            QMessageBox.warning(self, "No Image", "There is no aggregated image to save.")
            return

        default_path = "aggregated_image.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Aggregated Image", default_path,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff);;All Files (*)"
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.final_aggregated_image)
                self.log_message(f"Successfully saved image to: {file_path}")
            except Exception as e:
                error_msg = f"Failed to save image to {file_path}.\nError: {e}"
                self.log_message(f"ERROR: {error_msg}")
                QMessageBox.critical(self, "Save Error", error_msg)

    def closeEvent(self, event):
        self.log_message("--- Application closing, disconnecting from camera... ---")
        self.controller.disconnect()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraTestApp()
    window.show()
    sys.exit(app.exec())