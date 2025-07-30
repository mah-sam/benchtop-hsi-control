import sys
import os
import cv2
import numpy as np
import math
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QFormLayout, QTabWidget, QTextEdit, QSpinBox, QCheckBox,
    QSplitter, QRadioButton
)
from PyQt6.QtGui import QPixmap, QImage, QFont
from PyQt6.QtCore import Qt, pyqtSignal

# --- CORE ANALYSIS LOGIC ---

def detect_lines_logic(image_path, canny1, canny2, hough_thresh, min_len, max_gap):
    """
    Detects vertical lines in an image, robust to color/grayscale and 8/16-bit inputs.
    """
    original_img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if original_img is None:
        return None, None, None, None, f"Error: Could not load image at {image_path}"

    base_gray_8bit = None
    color_img_for_drawing = None

    if len(original_img.shape) == 3:  # Color image
        if original_img.dtype != np.uint8:
            color_img_for_drawing = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            color_img_for_drawing = original_img.copy()
        
        if color_img_for_drawing.shape[2] == 4:
            color_img_for_drawing = cv2.cvtColor(color_img_for_drawing, cv2.COLOR_BGRA2BGR)
        
        base_gray_8bit = cv2.cvtColor(color_img_for_drawing, cv2.COLOR_BGR2GRAY)

    elif len(original_img.shape) == 2:  # Grayscale image
        if original_img.dtype != np.uint8:
            base_gray_8bit = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            base_gray_8bit = original_img.copy()
        color_img_for_drawing = cv2.cvtColor(base_gray_8bit, cv2.COLOR_GRAY2BGR)
    else:
        return None, None, None, None, "Unsupported image format: Not a 2D or 3D image."

    high_contrast_gray = cv2.normalize(base_gray_8bit, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(high_contrast_gray, canny1, canny2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_thresh, minLineLength=min_len, maxLineGap=max_gap)

    if lines is None:
        return None, None, None, None, "No lines were detected. Try adjusting Canny or Hough parameters."

    vertical_angles, vertical_line_x_coords = [], []
    angle_min_deg, angle_max_deg = 75, 105

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = math.atan2(-(y2 - y1), (x2 - x1))
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0: angle_deg += 180
        if angle_min_deg < angle_deg < angle_max_deg:
            vertical_angles.append(angle_deg)
            vertical_line_x_coords.extend([x1, x2])
            cv2.line(color_img_for_drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not vertical_angles:
        return None, None, None, None, f"No lines passed the vertical filter (angle between {angle_min_deg}° and {angle_max_deg}°)."

    return base_gray_8bit, color_img_for_drawing, vertical_angles, vertical_line_x_coords, None
      
def apply_correction_logic(original_img, vertical_angles, vertical_line_x_coords):
    h, w = original_img.shape
    image_center = (w // 2, h // 2)

    average_angle_deg = np.mean(vertical_angles)
    average_line_x_pos = np.mean(vertical_line_x_coords)

    # --- Rotation ---
    rotation_angle_to_correct = 90.0 - average_angle_deg
    rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_angle_to_correct, 1.0)
    horizontal_shift = image_center[0] - average_line_x_pos
    translation_matrix = np.float32([[1, 0, horizontal_shift], [0, 1, 0]])
    rotated_img = cv2.warpAffine(original_img, rotation_matrix, (w, h))
    rotated_and_centered_img = cv2.warpAffine(rotated_img, translation_matrix, (w, h))

    # --- Shear ---
    average_angle_rad = np.deg2rad(average_angle_deg)
    tan_val = np.tan(average_angle_rad)
    slope_dx_dy = -1.0 / tan_val if abs(tan_val) > 1e-9 else 1e9
    sheared_img = np.zeros_like(original_img)
    for y in range(h):
        x_on_tilted_line = average_line_x_pos + (y - image_center[1]) * slope_dx_dy
        shift = average_line_x_pos - x_on_tilted_line
        row_data = original_img[y, :]
        int_shift = int(round(shift))
        if int_shift > 0:
            if int_shift < w: sheared_img[y, int_shift:] = row_data[:w - int_shift]
        elif int_shift < 0:
            abs_shift = abs(int_shift)
            if abs_shift < w: sheared_img[y, :w - abs_shift] = row_data[abs_shift:]
        else:
            sheared_img[y, :] = row_data
    final_translation_needed = image_center[0] - average_line_x_pos
    shear_centering_matrix = np.float32([[1, 0, final_translation_needed], [0, 1, 0]])
    sheared_and_centered_img = cv2.warpAffine(sheared_img, shear_centering_matrix, (w, h))

    results_data = {
        "analysis_summary": {"detected_lines": len(vertical_angles), "average_tilt_degrees": average_angle_deg, "average_x_position_pixels": average_line_x_pos},
        "rotation_correction": {"rotation_matrix": rotation_matrix.tolist(), "translation_matrix": translation_matrix.tolist()},
        "shear_correction": {"avg_angle_deg_for_calc": average_angle_deg, "avg_x_pos_for_calc": average_line_x_pos, "shear_centering_matrix": shear_centering_matrix.tolist()}
    }
    images = {"corrected_rotation": rotated_and_centered_img, "corrected_shear": sheared_and_centered_img}
    return images, results_data

# --- WIDGET CLASS ---
class ImageStraighteningWidget(QWidget):
    calibration_finished = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filepath = ""
        self.cv_images = {}
        self.analysis_results = {}
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0,0,0,0)
        control_panel = self._create_control_panel()
        visuals_panel = self._create_visuals_panel()
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.addWidget(control_panel)
        content_splitter.addWidget(visuals_panel)
        content_splitter.setSizes([350, 1050])
        top_layout.addWidget(content_splitter)
        log_panel = self._create_log_panel()
        main_splitter.addWidget(top_widget)
        main_splitter.addWidget(log_panel)
        main_splitter.setSizes([700, 200])
        final_layout = QVBoxLayout(self)
        final_layout.addWidget(main_splitter)
        self._update_ui_state()
        self.log_message("Component ready. Load an image to begin analysis.")

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
        
        params_group = QGroupBox("2. Analysis Parameters")
        params_layout = QFormLayout(params_group)
        self.param_inputs = {
            "Canny Thresh 1": QSpinBox(minimum=0, maximum=255, value=50),
            "Canny Thresh 2": QSpinBox(minimum=0, maximum=255, value=150),
            "Hough Thresh": QSpinBox(minimum=1, maximum=500, value=100),
            "Min Line Length": QSpinBox(minimum=10, maximum=2000, value=200),
            "Max Line Gap": QSpinBox(minimum=1, maximum=500, value=30)
        }
        for label, spinbox in self.param_inputs.items():
            params_layout.addRow(label, spinbox)
            spinbox.valueChanged.connect(self.on_param_change)
        
        method_group = QGroupBox("3. Correction Method")
        method_layout = QVBoxLayout(method_group)
        
        # ### MODIFIED ###: Changed radio button labels for clarity
        self.rotation_radio = QRadioButton("Rotation")
        self.shear_radio = QRadioButton("Shear")
        
        self.shear_radio.setChecked(True)
        
        self.apply_translation_checkbox = QCheckBox("Apply Centering (Translation)")
        self.apply_translation_checkbox.setChecked(True)
        
        method_layout.addWidget(self.rotation_radio)
        method_layout.addWidget(self.shear_radio)
        method_layout.addWidget(self.apply_translation_checkbox)

        actions_group = QGroupBox("4. Actions")
        actions_layout = QVBoxLayout(actions_group)
        self.auto_analyze_checkbox = QCheckBox("Auto-analyze on parameter change")
        self.auto_analyze_checkbox.setChecked(True)
        self.analyze_button = QPushButton("Analyze Now")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setFixedHeight(40)
        self.accept_button = QPushButton("Save")
        self.accept_button.clicked.connect(self._on_accept)
        self.accept_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        actions_layout.addWidget(self.auto_analyze_checkbox)
        actions_layout.addWidget(self.analyze_button)
        actions_layout.addWidget(self.accept_button)
        
        summary_group = QGroupBox("Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_label = QLabel("<i>Analysis has not been run.</i>")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        
        layout.addWidget(file_group)
        layout.addWidget(params_group)
        layout.addWidget(method_group)
        layout.addWidget(actions_group)
        layout.addWidget(summary_group)
        layout.addStretch()
        return panel

    def _create_visuals_panel(self):
        self.visuals_tabs = QTabWidget()
        detection_tab = QWidget()
        detection_layout = QHBoxLayout(detection_tab)
        self.input_image_label = self._create_image_display_label("Input Image")
        self.detected_image_label = self._create_image_display_label("Detected Lines")
        detection_layout.addWidget(self.input_image_label)
        detection_layout.addWidget(self.detected_image_label)
        correction_tab = QWidget()
        correction_layout = QHBoxLayout(correction_tab)
        self.rotated_image_label = self._create_image_display_label("Corrected (Rotation)")
        self.sheared_image_label = self._create_image_display_label("Corrected (Shear)")
        correction_layout.addWidget(self.rotated_image_label)
        correction_layout.addWidget(self.sheared_image_label)
        self.visuals_tabs.addTab(detection_tab, "Detection View")
        self.visuals_tabs.addTab(correction_tab, "Correction View")
        return self.visuals_tabs

    def _create_log_panel(self):
        log_tabs = QTabWidget()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.json_preview = QTextEdit()
        self.json_preview.setReadOnly(True)
        self.json_preview.setFont(QFont("Courier New", 9))
        log_tabs.addTab(self.log_text_edit, "Log")
        log_tabs.addTab(self.json_preview, "JSON Output")
        return log_tabs

    def _browse_for_file(self):
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", file_filter)
        if filepath:
            self.load_image(filepath)

    def load_image(self, filepath: str):
        if not os.path.exists(filepath):
            self.log_message(f"File not found: {filepath}", is_error=True)
            QMessageBox.critical(self, "Error", f"The provided image file does not exist:\n{filepath}")
            return
        self.filepath = filepath
        self.log_message(f"Loaded file: {filepath}")
        self.filepath_label.setText(f"<b>{os.path.basename(filepath)}</b>")
        self.cv_images.clear()
        self.analysis_results.clear()
        self.cv_images['input'] = cv2.imread(self.filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if self.cv_images['input'] is None:
            self.log_message(f"Failed to load image with OpenCV: {filepath}", is_error=True)
            self.filepath = ""
            self.filepath_label.setText("<i>Failed to load.</i>")
            return
        self._update_ui_state()
        self.run_analysis()
    
    def _on_accept(self):
        if not self.analysis_results:
            self.log_message("Accept clicked, but no valid analysis results are available.", is_error=True)
            QMessageBox.warning(self, "No Results", "Cannot save calibration because the analysis has not been run successfully.")
            return
        
        self.analysis_results["selected_method"] = "rotation" if self.rotation_radio.isChecked() else "shear"
        self.analysis_results["apply_translation"] = self.apply_translation_checkbox.isChecked()
            
        self.log_message(f"Calibration accepted. Method: '{self.analysis_results['selected_method']}', Centering: {self.analysis_results['apply_translation']}.")
        self.calibration_finished.emit(self.analysis_results)

    def on_param_change(self):
        if self.auto_analyze_checkbox.isChecked() and self.filepath:
            self.run_analysis()

    def run_analysis(self):
        if not self.filepath:
            QMessageBox.warning(self, "No Image", "Please load an image file first.")
            return
        self.log_message("Starting analysis...")
        self.setCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            params = {k: v.value() for k, v in self.param_inputs.items()}
            orig_gray, img_lines, angles, coords, error = detect_lines_logic(
                self.filepath, params["Canny Thresh 1"], params["Canny Thresh 2"],
                params["Hough Thresh"], params["Min Line Length"], params["Max Line Gap"]
            )
            if error:
                self.log_message(f"ERROR: {error}", is_error=True)
                self._clear_results()
                self._update_ui_state()
                return
            self.cv_images['detected'] = img_lines
            self.log_message(f"Line detection successful. Found {len(angles)} vertical lines.")
            corrected_images, results_data = apply_correction_logic(orig_gray, angles, coords)
            self.cv_images.update(corrected_images)
            self.analysis_results = results_data
            self.log_message("Image correction successful.")
        except Exception as e:
            self.log_message(f"An unexpected error occurred during analysis: {e}", is_error=True)
            self._clear_results()
        finally:
            self._update_display()
            self._update_ui_state()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_display(self):
        self.display_image(self.cv_images.get('input'), self.input_image_label, "Input Image")
        self.display_image(self.cv_images.get('detected'), self.detected_image_label, "Detected Lines")
        self.display_image(self.cv_images.get('corrected_rotation'), self.rotated_image_label, "Corrected (Rotation)")
        self.display_image(self.cv_images.get('corrected_shear'), self.sheared_image_label, "Corrected (Shear)")
        summary = self.analysis_results.get("analysis_summary")
        if summary:
            self.summary_label.setText(
                f"<b>Lines Detected:</b> {summary['detected_lines']}<br>"
                f"<b>Average Tilt:</b> {summary['average_tilt_degrees']:.2f}°<br>"
                f"<b>Average X-Position:</b> {summary['average_x_position_pixels']:.2f} px"
            )
        else:
            self.summary_label.setText("<i>Analysis failed or not run.</i>")
        if self.analysis_results:
            self.json_preview.setText(json.dumps(self.analysis_results, indent=4))
        else:
            self.json_preview.clear()

    def _update_ui_state(self):
        has_file = bool(self.filepath)
        has_results = bool(self.analysis_results)
        for spinbox in self.param_inputs.values():
            spinbox.setEnabled(has_file)
        self.analyze_button.setEnabled(has_file)
        self.auto_analyze_checkbox.setEnabled(has_file)
        self.accept_button.setEnabled(has_results)
        self.rotation_radio.setEnabled(has_file)
        self.shear_radio.setEnabled(has_file)
        self.apply_translation_checkbox.setEnabled(has_file)
        
    def _clear_results(self):
        keys_to_clear = ['detected', 'corrected_rotation', 'corrected_shear']
        for key in keys_to_clear:
            if key in self.cv_images:
                del self.cv_images[key]
        self.analysis_results.clear()
    
    def _create_image_display_label(self, title_text=""):
        label = QLabel(title_text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumSize(400, 300)
        label.setStyleSheet("QLabel { color: #888; font-weight: bold; font-size: 14px; border: 2px dashed #ccc; background-color: #f0f0f0; }")
        return label

    def display_image(self, cv_img, image_label, default_text=""):
        if cv_img is None:
            image_label.setText(default_text)
            image_label.setPixmap(QPixmap())
            return
        image_label.setStyleSheet("border: 1px solid #999;")
        display_img = cv_img
        if cv_img.dtype != np.uint8:
            self.log_message(f"Note: Converting {cv_img.dtype} image to 8-bit for display.")
            display_img = cv2.normalize(cv_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        h, w, *ch_info = display_img.shape
        if not ch_info:
            q_format = QImage.Format.Format_Grayscale8
            bytes_per_line = w
        else:
            ch = ch_info[0]
            q_format = QImage.Format.Format_BGR888 if ch == 3 else QImage.Format.Format_RGBA8888
            bytes_per_line = ch * w
        q_img = QImage(display_img.data, w, h, bytes_per_line, q_format)
        pixmap = QPixmap.fromImage(q_img)
        image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def log_message(self, message, is_error=False):
        if is_error:
            self.log_text_edit.append(f"<font color='red'><b>ERROR:</b> {message}</font>")
        else:
            self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()


class StandaloneRunner(QMainWindow):
    """A simple window to host and run the widget on its own."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone Image Straightening Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.straightening_widget = ImageStraighteningWidget()
        self.setCentralWidget(self.straightening_widget)
        self.straightening_widget.calibration_finished.connect(self.handle_calibration_complete)

    def handle_calibration_complete(self, calibration_data: dict):
        print("--- Standalone Mode: Calibration Data Received ---")
        print(json.dumps(calibration_data, indent=4))
        print("--------------------------------------------------")
        QMessageBox.information(
            self, 
            "Calibration Complete",
            "Calibration data has been printed to the console.\nThe application will now close."
        )
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StandaloneRunner()
    window.show()
    sys.exit(app.exec())