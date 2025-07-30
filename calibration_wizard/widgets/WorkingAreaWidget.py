import sys
import os
import cv2
import numpy as np
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QSlider, QSpinBox, QRadioButton
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, pyqtSignal

# --- CORE ALGORITHMS (Unchanged) ---
# ... (functions are identical, omitted for brevity) ...
def find_largest_inner_rect(binary_mask):
    if binary_mask is None or not np.any(binary_mask): return None
    h, w = binary_mask.shape; heights = np.zeros(w, dtype=int); max_area = 0; best_rect = (0, 0, 0, 0)
    for row_idx in range(h):
        for col_idx in range(w): heights[col_idx] = heights[col_idx] + 1 if binary_mask[row_idx, col_idx] else 0
        stack = [-1]
        for col_idx, height in enumerate(heights):
            while stack[-1] != -1 and heights[stack[-1]] >= height:
                h_pop = heights[stack.pop()]; w_pop = col_idx - stack[-1] - 1; area = h_pop * w_pop
                if area > max_area: max_area = area; best_rect = (stack[-1] + 1, row_idx - h_pop + 1, w_pop, h_pop)
            stack.append(col_idx)
        while stack[-1] != -1:
            h_pop = heights[stack.pop()]; w_pop = w - stack[-1] - 1; area = h_pop * w_pop
            if area > max_area: max_area = area; best_rect = (stack[-1] + 1, row_idx - h_pop + 1, w_pop, h_pop)
    return best_rect if max_area > 0 else None

def find_outer_bounding_box(binary_mask):
    if binary_mask is None or not np.any(binary_mask): return None
    rows, cols = np.where(binary_mask)
    x_min, x_max = np.min(cols), np.max(cols); y_min, y_max = np.min(rows), np.max(rows)
    width = x_max - x_min + 1; height = y_max - y_min + 1
    return (x_min, y_min, width, height)

# --- REFACTORED PYQT6 WIDGET CLASS ---

class WorkingAreaWidget(QWidget):
    # ... (The rest of the class is identical until _on_accept) ...
    calibration_finished = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filepath = ""
        self.original_image_gray = None
        self.last_display_image = None
        self.cropping_bbox = None
        main_layout = QHBoxLayout(self)
        control_panel = self._create_control_panel()
        self.image_label = self._create_image_display_label()
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.image_label, 1)
        self._update_ui_state()

    def _create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        panel.setFixedWidth(350)
        file_group = QGroupBox("1. Load Image")
        file_layout = QVBoxLayout(file_group)
        self.browse_button = QPushButton("Browse for Image...")
        self.browse_button.clicked.connect(self._browse_for_file)
        self.filepath_label = QLabel("<i>No file selected.</i>")
        self.filepath_label.setWordWrap(True)
        file_layout.addWidget(self.browse_button)
        file_layout.addWidget(self.filepath_label)
        analysis_group = QGroupBox("2. Analysis Controls")
        analysis_layout = QVBoxLayout(analysis_group)
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.inner_box_radio = QRadioButton("Inner Bounding Box (Safe Area)")
        self.outer_box_radio = QRadioButton("Outer Bounding Box (Total Extent)")
        self.inner_box_radio.setChecked(True)
        self.inner_box_radio.toggled.connect(self.run_analysis)
        self.outer_box_radio.toggled.connect(self.run_analysis)
        mode_layout.addWidget(self.inner_box_radio); mode_layout.addWidget(self.outer_box_radio)
        threshold_form_layout = QFormLayout()
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=255, value=200)
        self.threshold_slider.valueChanged.connect(self._slider_value_changed)
        self.threshold_spinbox = QSpinBox(minimum=0, maximum=255, value=200)
        self.threshold_spinbox.valueChanged.connect(self._spinbox_value_changed)
        self.analyze_button = QPushButton("Re-Analyze")
        self.analyze_button.setFixedHeight(30)
        self.analyze_button.clicked.connect(self.run_analysis)
        threshold_form_layout.addRow("Threshold:", self.threshold_slider)
        threshold_form_layout.addRow("", self.threshold_spinbox)
        analysis_layout.addWidget(mode_group); analysis_layout.addLayout(threshold_form_layout)
        analysis_layout.addWidget(self.analyze_button)
        results_group = QGroupBox("3. Results & Finalize")
        results_layout = QFormLayout(results_group)
        self.bbox_label = QLabel("<i>N/A</i>")
        self.accept_button = QPushButton("Save")
        self.accept_button.clicked.connect(self._on_accept)
        self.accept_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        results_layout.addRow("Bounding Box (x,y,w,h):", self.bbox_label)
        results_layout.addRow(self.accept_button)
        layout.addWidget(file_group)
        layout.addWidget(analysis_group)
        layout.addWidget(results_group)
        layout.addStretch()
        return panel

    def _create_image_display_label(self):
        label = QLabel("Load a white image to begin analysis")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(600, 400)
        label.setStyleSheet("QLabel { color: #888; font-weight: bold; border: 2px dashed #ccc; background-color: #f0f0f0; }")
        return label

    def _browse_for_file(self):
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Select White Image", "", file_filter)
        if filepath: self.load_image(filepath)

    def load_image(self, filepath: str):
        if not os.path.exists(filepath):
            QMessageBox.critical(self, "Error", f"The provided image file does not exist:\n{filepath}")
            return
        self.filepath = filepath
        self.filepath_label.setText(f"<b>{os.path.basename(filepath)}</b>")
        img = cv2.imread(self.filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            QMessageBox.critical(self, "Error", f"Could not load image file with OpenCV:\n{filepath}")
            self.filepath = ""
            return
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8: img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.original_image_gray = img
        self.cropping_bbox = None; self.last_display_image = None
        self.bbox_label.setText("<i>N/A</i>")
        self._display_cv_image(self.original_image_gray)
        self._update_ui_state()
        self.run_analysis()

    # ### MODIFIED ### - This is the corrected method
    def _on_accept(self):
        """
        Constructs the data dictionary with JSON-safe types and emits the completion signal.
        """
        if not self.cropping_bbox or self.original_image_gray is None:
            QMessageBox.warning(self, "No Data", "No valid bounding box to save.")
            return

        img_h, img_w = self.original_image_gray.shape
        x, y, w, h = self.cropping_bbox
        analysis_mode = "inner" if self.inner_box_radio.isChecked() else "outer"

        # Explicitly cast all NumPy numeric types to standard Python integers.
        # This is the fix for the 'not JSON serializable' TypeError.
        crop_data = {
            "source_file": self.filepath,
            "analysis_mode": analysis_mode,
            "image_dimensions": {"width": int(img_w), "height": int(img_h)},
            "threshold_value": self.threshold_slider.value(), # This is already a Python int
            "bbox_pixels": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
            "bbox_normalized": {
                "x_norm": round(int(x) / int(img_w), 6), 
                "y_norm": round(int(y) / int(img_h), 6),
                "width_norm": round(int(w) / int(img_w), 6), 
                "height_norm": round(int(h) / int(img_h), 6)
            }
        }
        self.calibration_finished.emit(crop_data)

    def _slider_value_changed(self, value):
        self.threshold_spinbox.blockSignals(True); self.threshold_spinbox.setValue(value)
        self.threshold_spinbox.blockSignals(False); self.run_analysis()

    def _spinbox_value_changed(self, value):
        self.threshold_slider.blockSignals(True); self.threshold_slider.setValue(value)
        self.threshold_slider.blockSignals(False); self.run_analysis()

    def run_analysis(self):
        if self.original_image_gray is None: return
        QApplication.processEvents()
        threshold = self.threshold_slider.value()
        valid_mask = (self.original_image_gray >= threshold).astype(np.uint8)
        if self.inner_box_radio.isChecked(): self.cropping_bbox = find_largest_inner_rect(valid_mask)
        else: self.cropping_bbox = find_outer_bounding_box(valid_mask)
        display_image = cv2.cvtColor(self.original_image_gray, cv2.COLOR_GRAY2BGR)
        red_overlay = np.zeros_like(display_image); red_overlay[valid_mask == 0] = [0, 0, 255]
        display_image = cv2.addWeighted(red_overlay, 0.4, display_image, 0.6, 0)
        if self.cropping_bbox:
            x, y, w, h = self.cropping_bbox
            cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            self.bbox_label.setText(f"({x}, {y}, {w}, {h})")
        else: self.bbox_label.setText("<i>No valid area found</i>")
        self.last_display_image = display_image
        self._display_cv_image(self.last_display_image)
        self._update_ui_state()

    def _update_ui_state(self):
        has_image = self.original_image_gray is not None
        has_bbox = self.cropping_bbox is not None
        self.inner_box_radio.setEnabled(has_image); self.outer_box_radio.setEnabled(has_image)
        self.threshold_slider.setEnabled(has_image); self.threshold_spinbox.setEnabled(has_image)
        self.analyze_button.setEnabled(has_image); self.accept_button.setEnabled(has_bbox)

    def _display_cv_image(self, cv_img):
        if cv_img is None:
            self.image_label.setText("Load a white image to begin analysis")
            self.image_label.setStyleSheet("QLabel { color: #888; font-weight: bold; border: 2px dashed #ccc; background-color: #f0f0f0; }")
            return
        self.image_label.setStyleSheet("border: 1px solid #999;")
        h, w, *ch = cv_img.shape
        bytes_per_line = w
        q_format = QImage.Format.Format_Grayscale8 if not ch else QImage.Format.Format_BGR888
        if ch: bytes_per_line = 3 * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, q_format)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_display_image is not None: self._display_cv_image(self.last_display_image)
        elif self.original_image_gray is not None: self._display_cv_image(self.original_image_gray)


class StandaloneRunner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone Working Area Finder")
        self.setGeometry(100, 100, 1200, 800)
        self.working_area_widget = WorkingAreaWidget()
        self.setCentralWidget(self.working_area_widget)
        self.working_area_widget.calibration_finished.connect(self.handle_calibration_complete)

    def handle_calibration_complete(self, calibration_data: dict):
        print("--- Standalone Mode: Working Area Data Received ---")
        print(json.dumps(calibration_data, indent=4))
        print("---------------------------------------------------")
        QMessageBox.information(
            self, "Calibration Complete",
            "Working area data has been printed to the console.\nThe application will now close.")
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StandaloneRunner()
    window.show()
    sys.exit(app.exec())