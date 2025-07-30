import sys
import os
import json
import cv2
import multiprocessing

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QMessageBox, QFileDialog, QInputDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

# Import the refactored widgets from their sub-folder
from .widgets.ImageStraighteningWidget import ImageStraighteningWidget
from .widgets.WorkingAreaWidget import WorkingAreaWidget
from .widgets.SpectralCalibratorWidget import SpectralCalibratorWidget

# --- ADD THIS BOILERPLATE AT THE VERY TOP ---
# This adds the project's root directory (hyperspectral_suite) to the Python path,
# allowing it to find the 'core' and 'hardware' packages.
# It finds the directory of the current file, then goes up one level.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- END BOILERPLATE ---


# ### MODIFIED ###: Import the new transformers module
from core import transformers

class CalibrationWizard(QMainWindow):
    """
    A step-by-step wizard to guide the user through a complete
    instrument calibration process, applying corrections sequentially.
    """
    def __init__(self, project_dir = project_root):
        super().__init__()
        self.setWindowTitle("Hyperspectral Calibration Wizard")
        self.setGeometry(100, 100, 1500, 950)
        self.current_step = 0
        self.project_dir = project_dir
        
        # ### MODIFIED ###: Added 'output_dimensions' to the data structure to store final image size.
        self.calibration_data = {
            "straightening": None, 
            "cropping": None, 
            "spectral": None, 
            "output_dimensions": None
        }
        
        self.image_paths = {"original_straighten": None, "corrected_straighten": None}
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        self.instruction_label = QLabel("Welcome!")
        font = QFont(); font.setPointSize(14); font.setBold(True)
        self.instruction_label.setFont(font)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction_label.setStyleSheet("padding: 10px; color: #FFFFFF;")
        main_layout.addWidget(self.instruction_label)
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget, 1)
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("<< Previous")
        self.next_button = QPushButton("Next >>")
        self.finish_button = QPushButton("Finish and Save All")
        self.prev_button.setFixedSize(150, 40)
        self.next_button.setFixedSize(150, 40)
        self.finish_button.setFixedSize(200, 40)
        self.finish_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.finish_button)
        main_layout.addLayout(nav_layout)
        self._setup_pages()
        self._connect_signals()
        self._update_ui_for_step()

    def _setup_pages(self):
        self.straightening_widget = ImageStraighteningWidget()
        self.working_area_widget = WorkingAreaWidget()
        self.spectral_widget = SpectralCalibratorWidget()
        self.stacked_widget.addWidget(self.straightening_widget)
        self.stacked_widget.addWidget(self.working_area_widget)
        self.stacked_widget.addWidget(self.spectral_widget)
    
    def _connect_signals(self):
        self.prev_button.clicked.connect(self._go_previous)
        self.next_button.clicked.connect(self._go_next)
        self.finish_button.clicked.connect(self._finish_calibration)
        self.straightening_widget.calibration_finished.connect(self._on_straighten_complete)
        self.working_area_widget.browse_button.clicked.disconnect()
        self.working_area_widget.browse_button.clicked.connect(self._load_and_straighten_for_crop_step)
        self.working_area_widget.calibration_finished.connect(self._on_crop_complete)
        self.spectral_widget.add_btn.clicked.disconnect()
        self.spectral_widget.add_btn.clicked.connect(self._load_and_correct_for_spectral_step)
        self.spectral_widget.calibration_finished.connect(self._on_spectral_complete)

    def _update_ui_for_step(self):
        self.stacked_widget.setCurrentIndex(self.current_step)
        instructions = [
            "Step 1 (Tilt): Load a RAW vertical line image. Adjust parameters, select a method, then click 'Save' to commit.",
            "Step 2 (Crop): Load a RAW saturated white image. The wizard will auto-apply the tilt correction. Adjust parameters, then click 'Save' to commit.",
            "Step 3 (Wavelength): Add RAW monochromatic line images. The wizard will auto-apply all corrections. Run calibration, then click 'Save' to commit."
        ]
        self.instruction_label.setText(instructions[self.current_step])
        self.prev_button.setEnabled(self.current_step > 0)
        self.next_button.setVisible(self.current_step < self.stacked_widget.count() - 1)
        self.finish_button.setVisible(self.current_step == self.stacked_widget.count() - 1)
        self.next_button.setEnabled(False) 
        self.finish_button.setEnabled(False)

    def _go_next(self):
        if self.current_step < self.stacked_widget.count() - 1: self.current_step += 1; self._update_ui_for_step()
    def _go_previous(self):
        if self.current_step > 0: self.current_step -= 1; self._update_ui_for_step()

    def _on_straighten_complete(self, data):
        print("Wizard: Step 1 (Straighten) complete."); self.calibration_data["straightening"] = data
        self.image_paths['original_straighten'] = self.straightening_widget.filepath; self.next_button.setEnabled(True)
        QMessageBox.information(self, "Step Complete", "Straightening calibration has been accepted.\nClick 'Next' to proceed.")
    
    def _on_crop_complete(self, data):
        """
        Handles the completion of the cropping step. Saves the cropping parameters
        and extracts the final output dimensions.
        """
        print("Wizard: Step 2 (Crop) complete.")
        # First, save the complete set of cropping parameters.
        self.calibration_data["cropping"] = data

        # ### MODIFIED ###: Access the nested dictionary to get the dimensions.
        # This now correctly handles the data structure you provided in the error log.
        try:
            # The data is nested, so we access it via 'bbox_pixels'
            crop_params = data['bbox_pixels']
            width = crop_params['width']
            height = crop_params['height']

            # Create the dimensions dictionary and save it
            dims = {"width": width, "height": height}
            self.calibration_data["output_dimensions"] = dims
            
            print(f"Wizard: Final output dimensions saved successfully: {dims}")
            self.next_button.setEnabled(True)
            QMessageBox.information(self, "Step Complete", "Cropping calibration has been accepted.\nClick 'Next' to proceed.")

        except (KeyError, TypeError) as e:
            # This robust error handling will catch issues if the data structure is ever unexpected.
            print(f"Warning: Could not determine output dimensions from cropping data. Error: {e}. Data received: {data}")
            self.next_button.setEnabled(False) # Keep 'Next' disabled
            QMessageBox.warning(self, "Data Error", "Could not save the output dimensions from the cropping step due to an unexpected data format. Please try cropping the image again.")

    def _on_spectral_complete(self, data):
        print("Wizard: Step 3 (Spectral) complete.")
        self.calibration_data["spectral"] = data
        self.finish_button.setEnabled(True)
        QMessageBox.information(self, "Final Step Complete", "Wavelength calibration has been accepted.\nClick 'Finish and Save All'.")
        
    def _finish_calibration(self):
        print("Wizard: Finishing calibration process.")
        try:
            assets_dir = os.path.join(self.project_dir, 'assets'); os.makedirs(assets_dir, exist_ok=True)
            save_path = os.path.join(assets_dir, "master_calibration.json")
            final_output = {
                "wizard_version": "1.0", "creation_date": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_steps": self.calibration_data
            }
            with open(save_path, 'w') as f: json.dump(final_output, f, indent=4)
            QMessageBox.information(self, "Success!", f"Master calibration file saved successfully to:\n{save_path}")
            self.close()
        except Exception as e: QMessageBox.critical(self, "Save Error", f"Could not save the master calibration file: {e}")

    def _load_and_straighten_for_crop_step(self):
        if not self.calibration_data.get("straightening"): QMessageBox.warning(self, "Prerequisite Missing", "Please complete Step 1 first."); return
        raw_path, _ = QFileDialog.getOpenFileName(self, "Select RAW White Image for Cropping", "", "Image Files (*.png *.tif *.tiff *.jpg)")
        if not raw_path: return
        try:
            print(f"Applying straightening to {os.path.basename(raw_path)} for crop step...")
            # ### MODIFIED ###: Call function from transformers module
            corrected_img = transformers.apply_straightening(raw_path, self.calibration_data["straightening"])
            assets_dir = os.path.join(os.path.dirname(__file__), "assets"); os.makedirs(assets_dir, exist_ok=True)
            temp_path = os.path.join(assets_dir, "temp_straightened_for_crop.png")
            cv2.imwrite(temp_path, corrected_img)
            self.working_area_widget.load_image(temp_path)
        except Exception as e: QMessageBox.critical(self, "Processing Error", f"Failed to apply straightening correction: {e}")

    def _load_and_correct_for_spectral_step(self):
        # ### MODIFIED ###: Added a check for output_dimensions.
        if not self.calibration_data.get("straightening") or not self.calibration_data.get("cropping") or not self.calibration_data.get("output_dimensions"):
            QMessageBox.warning(self, "Prerequisite Missing", "Please complete Step 1 and 2 first."); return
        
        raw_paths, _ = QFileDialog.getOpenFileNames(self, "Select RAW Monochromatic Images", "", "Image Files (*.png *.tif *.tiff *.jpg)")
        if not raw_paths: return
        
        assets_dir = os.path.join(os.path.dirname(__file__), "assets"); os.makedirs(assets_dir, exist_ok=True)
        processed_files_for_widget = []
        for raw_path in raw_paths:
            wavelength, ok = QInputDialog.getDouble(self, "Enter Wavelength", f"Enter wavelength (nm) for:\n{os.path.basename(raw_path)}", value=0.0, min=0.0, decimals=3)
            if not ok: continue
            try:
                print(f"Applying all corrections to {os.path.basename(raw_path)} for spectral step...")
                # ### MODIFIED ###: Call functions from transformers module
                straightened_img = transformers.apply_straightening(raw_path, self.calibration_data["straightening"])
                cropped_img = transformers.apply_cropping(straightened_img, self.calibration_data["cropping"])
                
                # ### ADDED ###: Verification check
                # Verify that the cropped image size matches the saved dimensions.
                h, w = cropped_img.shape[:2]
                saved_dims = self.calibration_data["output_dimensions"]
                if w != saved_dims['width'] or h != saved_dims['height']:
                    QMessageBox.critical(self, "Dimension Mismatch", f"The processed image dimensions ({w}x{h}) do not match the saved calibration dimensions ({saved_dims['width']}x{saved_dims['height']}). Aborting.")
                    return

                base_name, ext = os.path.splitext(os.path.basename(raw_path))
                temp_path = os.path.join(assets_dir, f"temp_{base_name}_corrected.png")
                cv2.imwrite(temp_path, cropped_img)
                processed_files_for_widget.append((temp_path, wavelength))
            except Exception as e: QMessageBox.critical(self, "Processing Error", f"Failed to process {os.path.basename(raw_path)}: {e}"); return
        self.spectral_widget.load_files(processed_files_for_widget)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Exit Wizard', "Are you sure you want to exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            print("Wizard: Closing and stopping all threads.")
            self.spectral_widget.stop_all_threads()
            event.accept()
        else: event.ignore()

    def stop_all_threads(self):
        """
        Safely stops the running calibration thread.
        This method is now robust against being called during application shutdown.
        """
        # ### MODIFIED ###: Wrap the entire logic in a try/except block.
        try:
            # Check if the thread object exists and is currently running.
            # This check can fail with a RuntimeError if the underlying C++ object
            # has already been deleted by Qt's garbage collector during shutdown.
            if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
                print("SpectralCalibratorWidget: Requesting thread to stop...")
                
                # Assuming your worker has a 'stop' method or a flag
                if hasattr(self, 'worker') and self.worker:
                    self.worker.stop() # Tell the worker loop to finish

                self.thread.quit()  # Ask the event loop to exit
                
                # Wait for the thread to finish cleanly.
                # A timeout prevents the application from hanging if the thread is stuck.
                if not self.thread.wait(3000): # Wait up to 3 seconds
                    print("SpectralCalibratorWidget: Thread did not stop in time. Terminating.")
                    self.thread.terminate() # Forcefully stop if it doesn't respond
                    self.thread.wait() # Wait for termination to complete

                print("SpectralCalibratorWidget: Thread stopped.")

        except RuntimeError:
            # This error is expected if the application is closing, as the C++ part
            # of the QThread object might have been deleted before this code runs.
            # We can safely ignore it and just log that it happened.
            print("SpectralCalibratorWidget: Thread was already deleted (expected during shutdown).")
        
        finally:
            # It's good practice to nullify the references after stopping.
            self.thread = None
            self.worker = None

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    wizard = CalibrationWizard()
    wizard.show()
    sys.exit(app.exec())