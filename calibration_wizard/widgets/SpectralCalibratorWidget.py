import sys
import os
import time
import json
import queue
import threading
import multiprocessing
import numpy as np
from scipy.optimize import curve_fit
from PIL import Image

# Import PyQt6 components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QListWidget, QLineEdit, QSpinBox, QProgressBar, QLabel,
    QGroupBox, QSplitter, QTabWidget, QFileDialog, QInputDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal

# Import Matplotlib components for PyQt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# --- Core Scientific Functions & Worker (Unchanged) ---
# The _fit_one_column, find_line_centers_parallel, and Worker class
# remain exactly the same. They are omitted here for brevity.

def _fit_one_column(args):
    """Fits a Gaussian to a single column. Designed to be run in a parallel process."""
    column_index, profile, y_data, initial_sigma_guess, intensity_threshold = args
    def gaussian(x, a, mu, sigma, c): return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c
    if np.max(profile) < intensity_threshold: return column_index, np.nan
    try:
        height = len(profile); c_guess = np.min(profile); a_guess = np.max(profile) - c_guess; mu_guess = np.argmax(profile)
        p0 = [a_guess, mu_guess, initial_sigma_guess, c_guess]; bounds = ([0, 0, 0.1, 0], [np.inf, height, height*2, np.inf])
        popt, _ = curve_fit(gaussian, y_data, profile, p0=p0, bounds=bounds, maxfev=1000)
        return column_index, popt[1]
    except (RuntimeError, ValueError):
        if np.sum(profile) > 0: return column_index, np.sum(y_data * profile) / np.sum(profile)
        return column_index, np.nan
    except Exception: return column_index, np.nan

def find_line_centers_parallel(image_data, initial_sigma_guess=5.0, intensity_threshold=20, fit_stride=1, progress_queue=None):
    height, width = image_data.shape; y_data = np.arange(height)
    if fit_stride > width: fit_stride = max(1, width // 10)
    columns_to_fit_indices = np.arange(0, width, fit_stride)
    tasks = [(i, image_data[:, i], y_data, initial_sigma_guess, intensity_threshold) for i in columns_to_fit_indices]
    num_workers = max(1, os.cpu_count() - 1); results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        total_tasks = len(tasks)
        for i, result in enumerate(pool.imap_unordered(_fit_one_column, tasks)):
            results.append(result)
            if progress_queue: progress_queue.put((i + 1) / total_tasks * 100)
    results.sort(key=lambda x: x[0])
    fitted_indices = np.array([res[0] for res in results]); fitted_centers = np.array([res[1] for res in results])
    valid_mask = ~np.isnan(fitted_centers)
    if np.sum(valid_mask) < 2:
        final_centers = np.full(width, np.nan)
        if np.sum(valid_mask) == 1: final_centers[:] = fitted_centers[valid_mask][0]
        return final_centers
    final_centers = np.interp(np.arange(width), fitted_indices[valid_mask], fitted_centers[valid_mask])
    return final_centers

class Worker(QObject):
    overall_progress = pyqtSignal(int, str); detail_progress = pyqtSignal(int)
    finished = pyqtSignal(dict); error = pyqtSignal(str)
    def __init__(self, calibration_files, params):
        super().__init__(); self.calibration_files = calibration_files; self.params = params; self.is_running = True
    def run(self):
        try:
            line_centers_result = {}; pixel_centers = []; wavelengths = []
            try: img_h = np.array(Image.open(self.calibration_files[0][0]).convert('L')).shape[0]
            except FileNotFoundError: raise FileNotFoundError(f"Cannot open initial image file: {self.calibration_files[0][0]}")
            first_w, last_w = self.calibration_files[0][1], self.calibration_files[-1][1]
            dispersion_est = (last_w - first_w) / img_h if img_h > 0 else 0.1
            sigma_guess_px = (self.params['bandwidth_nm'] / dispersion_est) / 2.355 if dispersion_est > 0 else 5.0
            num_files = len(self.calibration_files); manager = multiprocessing.Manager(); progress_q = manager.Queue()
            progress_thread = threading.Thread(target=self._progress_monitor, args=(progress_q,), daemon=True); progress_thread.start()
            for i, (path, wavelength) in enumerate(self.calibration_files):
                if not self.is_running: break
                self.overall_progress.emit(int((i / num_files) * 100), f"Processing {os.path.basename(path)} ({i+1}/{num_files})...")
                self.detail_progress.emit(0)
                image = np.array(Image.open(path).convert('L'), dtype=np.float32)
                centers = find_line_centers_parallel(image, sigma_guess_px, self.params['fit_threshold'], self.params['fit_stride'], progress_q)
                line_centers_result[path] = centers; mid_spatial_idx = image.shape[1] // 2
                pixel_centers.append(centers[mid_spatial_idx]); wavelengths.append(wavelength)
            progress_q.put(None); progress_thread.join()
            self.overall_progress.emit(95, "Fitting polynomial...")
            pixel_centers, wavelengths = np.array(pixel_centers), np.array(wavelengths)
            valid_mask = ~np.isnan(pixel_centers)
            if np.sum(valid_mask) < self.params['poly_degree'] + 1: raise ValueError("Not enough valid points for polynomial fit.")
            poly_coeffs_result = np.polyfit(pixel_centers[valid_mask], wavelengths[valid_mask], self.params['poly_degree'])
            result_data = {"line_centers": line_centers_result, "poly_coeffs": poly_coeffs_result}
            self.finished.emit(result_data)
        except Exception as e: self.error.emit(str(e))
    def _progress_monitor(self, q):
        while True:
            try:
                progress = q.get()
                if progress is None: break
                self.detail_progress.emit(int(progress))
            except (EOFError, BrokenPipeError): break
    def stop(self): self.is_running = False

# --- REFACTORED Main Application Widget ---

class SpectralCalibratorWidget(QWidget):
    """A widget for performing spectral calibration. Can be run standalone or in a wizard."""
    calibration_finished = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration_files = []
        self.line_center_data = {}
        self.poly_coeffs = None
        self.thread = None
        self.worker = None
        self.all_controls = []
        self._create_widgets()
        self._connect_signals()
        
    def _create_widgets(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        splitter.addWidget(control_panel)

        file_group = QGroupBox("1. Input Files")
        file_layout = QVBoxLayout(file_group)
        self.file_listbox = QListWidget()
        file_layout.addWidget(self.file_listbox)
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("Add Images...")
        self.remove_btn = QPushButton("Remove Selected")
        btn_layout.addWidget(self.add_btn); btn_layout.addWidget(self.remove_btn)
        file_layout.addLayout(btn_layout)
        control_layout.addWidget(file_group)
        
        param_group = QGroupBox("2. Calibration Parameters")
        param_layout = QGridLayout(param_group)
        self.bandwidth_nm_edit = QLineEdit("10.0")
        self.poly_degree_spinbox = QSpinBox(minimum=1, maximum=5, value=2)
        self.fit_threshold_edit = QLineEdit("20")
        self.fit_stride_spinbox = QSpinBox(minimum=1, maximum=50, value=5)
        param_layout.addWidget(QLabel("Bandwidth (FWHM, nm):"), 0, 0); param_layout.addWidget(self.bandwidth_nm_edit, 0, 1)
        param_layout.addWidget(QLabel("Polynomial Degree:"), 1, 0); param_layout.addWidget(self.poly_degree_spinbox, 1, 1)
        param_layout.addWidget(QLabel("Intensity Threshold (0-255):"), 2, 0); param_layout.addWidget(self.fit_threshold_edit, 2, 1)
        param_layout.addWidget(QLabel("Fit Stride (1=all, >1=sample):"), 3, 0); param_layout.addWidget(self.fit_stride_spinbox, 3, 1)
        control_layout.addWidget(param_group)

        action_group = QGroupBox("3. Actions")
        action_layout = QVBoxLayout(action_group)
        self.run_button = QPushButton("RUN CALIBRATION")
        self.run_button.setStyleSheet("font-weight: bold; color: white; background-color: #003366; border: 1px solid #001a33; padding: 5px; border-radius: 3px;")
        
        # ### MODIFIED ###: Replaced Save button with Save and Exit
        self.accept_button = QPushButton("Save")
        self.accept_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.accept_button)
        control_layout.addWidget(action_group)
        
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.status_label = QLabel("Ready.")
        self.detail_progress_bar = QProgressBar()
        self.overall_progress_bar = QProgressBar()
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(QLabel("Detail Progress (Current Image):")); progress_layout.addWidget(self.detail_progress_bar)
        progress_layout.addWidget(QLabel("Overall Progress:")); progress_layout.addWidget(self.overall_progress_bar)
        control_layout.addWidget(progress_group)

        self.all_controls = [self.add_btn, self.remove_btn, self.bandwidth_nm_edit,
                             self.poly_degree_spinbox, self.fit_threshold_edit,
                             self.fit_stride_spinbox, self.accept_button, self.file_listbox, self.run_button]

        vis_panel = QWidget()
        vis_layout = QVBoxLayout(vis_panel)
        self.notebook = QTabWidget()
        vis_layout.addWidget(self.notebook)
        splitter.addWidget(vis_panel)
        
        self.fig_detection = plt.Figure(); self.ax_detection = self.fig_detection.add_subplot(111); self._create_tab("Line Detection", self.fig_detection)
        self.fig_fit = plt.Figure(); self.ax_fit = self.fig_fit.add_subplot(111); self._create_tab("1D Gaussian Fit", self.fig_fit)
        self.fig_map = plt.Figure(); self.ax_map = self.fig_map.add_subplot(111); self._create_tab("Wavelength Map", self.fig_map)
        
        splitter.setSizes([400, 1000])

    def _create_tab(self, title, figure):
        tab = QWidget(); layout = QVBoxLayout(tab)
        canvas = FigureCanvas(figure); toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar); layout.addWidget(canvas)
        self.notebook.addTab(tab, title)
        canvas.mpl_connect('button_press_event', lambda event: self._on_plot_click(event, title))

    def _connect_signals(self):
        self.add_btn.clicked.connect(self._add_files)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.run_button.clicked.connect(self._start_calibration)
        self.accept_button.clicked.connect(self._on_accept) # Changed from save to accept
        self.file_listbox.currentItemChanged.connect(self._on_listbox_select)

    def _toggle_controls(self, enabled):
        for control in self.all_controls:
            control.setEnabled(enabled)

    # --- Public API for Wizard ---
    def load_files(self, files_with_wavelengths: list):
        """Public method for a wizard to load files and their wavelengths."""
        self.calibration_files = files_with_wavelengths
        self._update_listbox()

    # --- Internal Logic ---
    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Calibration Images", "", "Image Files (*.png *.jpg *.jpeg *.tif *.tiff);;All files (*.*)")
        if not paths: return
        for path in paths:
            if any(p == path for p, w in self.calibration_files): continue
            wavelength, ok = QInputDialog.getDouble(self, "Enter Wavelength", f"Enter center wavelength (nm) for:\n{os.path.basename(path)}", value=0.0, min=0.0, decimals=3)
            if ok: self.calibration_files.append((path, wavelength))
        self._update_listbox()

    def _remove_selected(self):
        selected_items = self.file_listbox.selectedItems()
        if not selected_items: return
        selected_paths = {item.text().split('  (')[0] for item in selected_items}
        self.calibration_files = [(p, w) for p, w in self.calibration_files if os.path.basename(p) not in selected_paths]
        self._update_listbox()

    def _update_listbox(self):
        self.file_listbox.clear()
        self.calibration_files.sort(key=lambda x: x[1])
        for path, wavelength in self.calibration_files:
            self.file_listbox.addItem(f"{os.path.basename(path)}  ({wavelength} nm)")
            
    def _start_calibration(self):
        try:
            params = {
                'poly_degree': self.poly_degree_spinbox.value(),
                'bandwidth_nm': float(self.bandwidth_nm_edit.text()),
                'fit_threshold': int(self.fit_threshold_edit.text()),
                'fit_stride': self.fit_stride_spinbox.value()
            }
        except ValueError as e:
            QMessageBox.critical(self, "Parameter Error", f"Invalid parameter value: {e}")
            return
        if len(self.calibration_files) < params['poly_degree'] + 1:
            QMessageBox.critical(self, "Error", f"Need at least {params['poly_degree'] + 1} files for a degree {params['poly_degree']} polynomial fit.")
            return
        self._toggle_controls(False)
        self.overall_progress_bar.setValue(0); self.detail_progress_bar.setValue(0)
        self.status_label.setText("Starting calibration...")
        self.thread = QThread(); self.worker = Worker(self.calibration_files, params)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_calibration_complete)
        self.worker.error.connect(self._on_calibration_error)
        self.worker.overall_progress.connect(self._update_overall_progress)
        self.worker.detail_progress.connect(self.detail_progress_bar.setValue)
        self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _update_overall_progress(self, value, text):
        self.overall_progress_bar.setValue(value); self.status_label.setText(text)

    def _on_calibration_complete(self, result):
        self.line_center_data = result["line_centers"]
        self.poly_coeffs = result["poly_coeffs"]
        self._toggle_controls(True)
        self.overall_progress_bar.setValue(100); self.detail_progress_bar.setValue(100)
        self.status_label.setText("Calibration complete.")
        self._update_plots()
        QMessageBox.information(self, "Success", "Calibration complete. Check visualization tabs.")
    
    def _on_calibration_error(self, error_message):
        self._toggle_controls(True)
        self.status_label.setText(f"Error: {error_message}")
        self.overall_progress_bar.setValue(0); self.detail_progress_bar.setValue(0)
        QMessageBox.critical(self, "Calibration Error", error_message)

    def _on_accept(self):
        """Packages the calibration data and emits it via a signal."""
        if self.poly_coeffs is None:
            QMessageBox.critical(self, "Error", "No calibration has been run yet.")
            return
        
        calibration_map = {
            "description": "Wavelength map. Wavelength = f(pixel_row).",
            "poly_degree": self.poly_degree_spinbox.value(),
            "coefficients": self.poly_coeffs.tolist(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameters": {
                "bandwidth_nm": float(self.bandwidth_nm_edit.text()),
                "fit_threshold": int(self.fit_threshold_edit.text()),
                "fit_stride": self.fit_stride_spinbox.value()
            }
        }
        self.calibration_finished.emit(calibration_map)
    
    def stop_all_threads(self):
        """Public method to safely stop the worker thread if it's running."""
        if self.thread and self.thread.isRunning():
            self.worker.stop()
            self.thread.quit()
            self.thread.wait() # Wait for thread to finish
            
    # --- Plotting Methods (Unchanged) ---
    def _update_plots(self):
        selected_item = self.file_listbox.currentItem()
        if not selected_item: return
        base_name = selected_item.text().split('  (')[0]
        selected_path = next((p for p, w in self.calibration_files if os.path.basename(p) == base_name), None)
        if not selected_path: return
        if self.line_center_data:
            self._plot_line_detection(selected_path)
            img_width = np.array(Image.open(selected_path).convert('L')).shape[1]
            self._plot_1d_fit(selected_path, img_width // 2)
        if self.poly_coeffs is not None: self._plot_wavelength_map()
    def _on_listbox_select(self, current_item, previous_item): self._update_plots()
    def _on_plot_click(self, event, title):
        if title != "Line Detection" or event.inaxes != self.ax_detection or event.xdata is None: return
        col_idx = int(event.xdata)
        selected_item = self.file_listbox.currentItem()
        if not selected_item: return
        base_name = selected_item.text().split('  (')[0]
        selected_path = next((p for p,w in self.calibration_files if os.path.basename(p) == base_name), None)
        if selected_path: self._plot_1d_fit(selected_path, col_idx); self.notebook.setCurrentIndex(1)
    def _plot_line_detection(self, image_path):
        self.ax_detection.clear()
        try:
            image = np.array(Image.open(image_path).convert('L')); centers = self.line_center_data.get(image_path)
            self.ax_detection.imshow(image, cmap='gray', aspect='auto')
            if centers is not None:
                valid_idx = ~np.isnan(centers)
                self.ax_detection.plot(np.arange(len(centers))[valid_idx], centers[valid_idx], 'r.', markersize=2, label='Detected Centers')
                self.ax_detection.legend()
            self.ax_detection.set_title(f"Line Detection for {os.path.basename(image_path)}\n(Click a column to see 1D fit)")
            self.ax_detection.set_xlabel("Spatial Axis (pixel column)"); self.ax_detection.set_ylabel("Spectral Axis (pixel row)")
            self.fig_detection.canvas.draw()
        except Exception as e:
            self.ax_detection.text(0.5, 0.5, f"Error plotting:\n{e}", ha='center', va='center', color='red', transform=self.ax_detection.transAxes)
            self.fig_detection.canvas.draw()
    def _plot_1d_fit(self, image_path, column_index):
        self.ax_fit.clear()
        try:
            image = np.array(Image.open(image_path).convert('L'), dtype=np.float32); height, width = image.shape
            if not (0 <= column_index < width): return
            profile = image[:, column_index]; y_data = np.arange(height)
            center_val = self.line_center_data.get(image_path, [np.nan]*width)[column_index]
            self.ax_fit.plot(y_data, profile, 'b.', label='Raw Data')
            if np.isnan(center_val): self.ax_fit.text(0.5, 0.5, "No fit performed", ha='center', transform=self.ax_fit.transAxes, color='red')
            else: self.ax_fit.axvline(center_val, color='g', linestyle='--', label=f'Interpolated Center: {center_val:.2f} px')
            self.ax_fit.legend(); self.ax_fit.set_title(f"1D Profile for Column {column_index}")
            self.ax_fit.set_xlabel("Spectral Axis (pixel row)"); self.ax_fit.set_ylabel("Intensity (DN)"); self.ax_fit.grid(True)
            self.fig_fit.canvas.draw()
        except Exception as e:
            self.ax_fit.text(0.5, 0.5, f"Error plotting:\n{e}", ha='center', va='center', color='red', transform=self.ax_fit.transAxes)
            self.fig_fit.canvas.draw()
    def _plot_wavelength_map(self):
        self.ax_map.clear(); pixel_centers, wavelengths = [], []
        for path, w in self.calibration_files:
            try:
                mid_idx = np.array(Image.open(path).convert('L')).shape[1] // 2
                center_val = self.line_center_data[path][mid_idx]
                if not np.isnan(center_val): pixel_centers.append(center_val); wavelengths.append(w)
            except Exception: continue
        pixel_centers, wavelengths = np.array(pixel_centers), np.array(wavelengths)
        if len(pixel_centers) < self.poly_degree_spinbox.value() + 1: return
        self.ax_map.plot(pixel_centers, wavelengths, 'bo', label='Data Points (Spatial Center)')
        p_fit = np.poly1d(self.poly_coeffs); p_range = np.linspace(min(pixel_centers), max(pixel_centers), 200)
        self.ax_map.plot(p_range, p_fit(p_range), 'r-', label=f'Degree {self.poly_degree_spinbox.value()} Poly Fit')
        residuals = wavelengths - p_fit(pixel_centers); ax2 = self.ax_map.twinx()
        ax2.plot(pixel_centers, residuals, 'g--', marker='x', label='Residuals (nm)'); ax2.set_ylabel("Residuals (nm)", color='g')
        ax2.tick_params(axis='y', labelcolor='g'); ax2.axhline(0, color='g', linestyle=':', linewidth=0.8)
        lines, labels = self.ax_map.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best'); self.ax_map.set_title("Wavelength vs. Pixel Position (Wavelength Map)")
        self.ax_map.set_xlabel("Spectral Axis (pixel row)"); self.ax_map.set_ylabel("Wavelength (nm)")
        self.ax_map.grid(True); self.ax_map.legend_ = None; self.fig_map.tight_layout(); self.fig_map.canvas.draw()

# --- STANDALONE RUNNER ---
class StandaloneRunner(QMainWindow):
    """A simple window to host and run the widget on its own."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Standalone Spectral Calibrator")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet("QMainWindow { background-color: #444444; }") # Simple background

        self.calibrator_widget = SpectralCalibratorWidget()
        self.setCentralWidget(self.calibrator_widget)

        self.calibrator_widget.calibration_finished.connect(self.handle_calibration_complete)

    def handle_calibration_complete(self, calibration_data: dict):
        """Defines what happens when 'Save and Exit' is clicked in standalone mode."""
        print("--- Standalone Mode: Spectral Calibration Data Received ---")
        print(json.dumps(calibration_data, indent=4))
        print("---------------------------------------------------------")
        
        reply = QMessageBox.question(
            self, 
            "Calibration Complete",
            "Calibration data has been printed to the console.\n\nWould you like to save this to a file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            path, _ = QFileDialog.getSaveFileName(self, "Save Calibration Map", "", "JSON Files (*.json)")
            if path:
                try:
                    with open(path, 'w') as f:
                        json.dump(calibration_data, f, indent=4)
                    QMessageBox.information(self, "Saved", f"Calibration map saved to {path}")
                except Exception as e:
                    QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")

        self.close()

    def closeEvent(self, event):
        """Ensure the hosted widget's threads are properly stopped on exit."""
        self.calibrator_widget.stop_all_threads()
        event.accept()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = StandaloneRunner()
    window.show()
    sys.exit(app.exec())