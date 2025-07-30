# """
# Module for controlling a FLIR (formerly Point Grey) camera using the PySpin SDK.

# This module provides the CameraController class, which abstracts the SDK calls
# for camera initialization, configuration, live view, and threaded bulk acquisition.
# """

# import os
# import sys
# import numpy as np
# from PySpin.PySpin import System, CameraPtr, SpinnakerException, CStringPtr, CEnumerationPtr, IsReadable, IsWritable, CFloatPtr

# from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer


# class AcquisitionThread(QThread):
#     # ... (No changes in this class) ...
#     finished = pyqtSignal(str)
#     progress = pyqtSignal(int, int)
#     error = pyqtSignal(str)
#     def __init__(self, cam: CameraPtr, num_images: int, save_dir: str, parent=None):
#         super().__init__(parent)
#         self.cam = cam
#         self.num_images = num_images
#         self.save_dir = save_dir
#     def run(self):
#         try:
#             os.makedirs(self.save_dir, exist_ok=True)
#             nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
#             node_device_serial_number = CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
#             device_serial_number = node_device_serial_number.GetValue() if IsReadable(node_device_serial_number) else ''
#             for i in range(self.num_images):
#                 image_result = self.cam.GetNextImage(1000)
#                 if image_result.IsIncomplete():
#                     self.error.emit(f'Image {i} incomplete with status {image_result.GetImageStatus()}')
#                 else:
#                     filename = f'Acquisition-{device_serial_number}-{i}.jpg' if device_serial_number else f'Acquisition-{i}.jpg'
#                     full_path = os.path.join(self.save_dir, filename)
#                     image_result.Save(full_path)
#                     self.progress.emit(i + 1, self.num_images)
#                 image_result.Release()
#             self.finished.emit(self.save_dir)
#         except SpinnakerException as ex:
#             self.error.emit(f"Acquisition Error: {ex}")


# class CameraController(QObject):
#     # ... (No changes to signals) ...
#     status_update = pyqtSignal(str)
#     connection_lost = pyqtSignal(str)
#     new_live_frame = pyqtSignal(object)
#     acquisition_finished = pyqtSignal(str)
#     acquisition_progress = pyqtSignal(int, int)
#     aggregation_finished = pyqtSignal(object) # Emits the final aggregated NumPy array
#     aggregation_updated = pyqtSignal(object)

#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self.system = None
#         self.cam = None
#         # --- FIX: Add cam_list as an instance variable ---
#         self.cam_list = None
#         self.nodemap = None
#         self.is_connected = False
#         self.is_acquiring = False
#         self.acquisition_thread = None
#         self._live_view_timer = QTimer(self)
#         self._live_view_timer.timeout.connect(self._capture_live_frame)
#         self.is_aggregating = False
#         self.aggregated_image = None
#         self.aggregation_frame_count = 0
#         self.aggregation_target_frames = 0
#         self.aggregation_threshold = 20 # Default threshold
        

#     # ... (set_exposure_time method is unchanged) ...
#     def set_exposure_time(self, exposure_us: float):
#         if not self.is_connected:
#             self.status_update.emit("Error: Cannot set exposure. Camera not connected.")
#             return False
#         try:
#             node_exposure_auto = CEnumerationPtr(self.nodemap.GetNode('ExposureAuto'))
#             if IsWritable(node_exposure_auto):
#                 entry_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
#                 node_exposure_auto.SetIntValue(entry_exposure_auto_off.GetValue())
#                 self.status_update.emit("Automatic exposure turned OFF.")
#             node_exposure_time = CFloatPtr(self.nodemap.GetNode('ExposureTime'))
#             if IsWritable(node_exposure_time):
#                 min_val = node_exposure_time.GetMin()
#                 max_val = node_exposure_time.GetMax()
#                 value_to_set = min(max(exposure_us, min_val), max_val)
#                 node_exposure_time.SetValue(value_to_set)
#                 self.status_update.emit(f"Exposure time set to {value_to_set:.2f} us.")
#                 return True
#             else:
#                 self.status_update.emit("Error: Exposure time node is not writable.")
#                 return False
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Error setting exposure: {ex}")
#             return False

#     def connect(self):
#         if self.is_connected: return True
#         try:
#             self.system = System.GetInstance()
#             # --- FIX: Store cam_list in self ---
#             self.cam_list = self.system.GetCameras()

#             if self.cam_list.GetSize() == 0:
#                 self.status_update.emit("Error: No cameras detected.")
#                 # --- FIX: Clear the list before releasing the system ---
#                 self.cam_list.Clear()
#                 self.system.ReleaseInstance()
#                 self.system = None
#                 return False

#             self.cam = self.cam_list.GetByIndex(0)
#             self.cam.Init()
#             self.nodemap = self.cam.GetNodeMap()
#             if not self._configure_camera():
#                 self.cam.DeInit()
#                 del self.cam
#                 return False
#             self.is_connected = True
#             self.status_update.emit("Camera connected and configured successfully.")
#             return True
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Camera Connection Error: {ex}")
#             return False

#     def disconnect(self):
#         if not self.is_connected: return
#         if self.is_acquiring: self.stop_live_view()

#         # --- FIX: DeInit and delete the camera object first ---
#         if self.cam:
#             try:
#                 self.cam.DeInit()
#             except SpinnakerException as ex:
#                 self.status_update.emit(f"Error during DeInit: {ex}")
#             del self.cam
#             self.cam = None
        
#         # --- FIX: Clear the camera list next ---
#         if self.cam_list:
#             try:
#                 self.cam_list.Clear()
#             except SpinnakerException as ex:
#                 self.status_update.emit(f"Error clearing camera list: {ex}")
#             self.cam_list = None
        
#         # --- FIX: Release the system instance LAST ---
#         if self.system:
#             try:
#                 self.system.ReleaseInstance()
#             except SpinnakerException as ex:
#                 self.status_update.emit(f"Error releasing system instance: {ex}")
#             self.system = None
        
#         self.is_connected = False
#         self.status_update.emit("Camera disconnected.")

#     # --- The rest of the file is unchanged ---
#     def _configure_camera(self):
#         try:
#             node_acq_mode = CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
#             node_acq_mode_continuous = node_acq_mode.GetEntryByName('Continuous')
#             node_acq_mode.SetIntValue(node_acq_mode_continuous.GetValue())
#             self.status_update.emit("Acquisition mode set to Continuous.")
#             node_pixel_format = CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
#             node_pixel_format_mono8 = node_pixel_format.GetEntryByName('Mono8')
#             node_pixel_format.SetIntValue(node_pixel_format_mono8.GetValue())
#             self.status_update.emit("Pixel format set to Mono8.")
#             return True
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Configuration Error: {ex}")
#             return False

#     def start_live_view(self, interval_ms: int = 33):
#         if not self.is_connected or self.is_acquiring: return
#         try:
#             self.is_acquiring = True
#             self.cam.BeginAcquisition()
#             self._live_view_timer.setInterval(interval_ms)
#             self._live_view_timer.start()
#             self.status_update.emit("Live view started.")
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Live View Error: {ex}")
#             self.is_acquiring = False

#     def stop_live_view(self):
#         self._live_view_timer.stop()
#         if self.is_acquiring:
#             try:
#                 self.cam.EndAcquisition()
#                 self.is_acquiring = False
#                 self.status_update.emit("Live view stopped.")
#             except SpinnakerException as ex:
#                 self.status_update.emit(f"Error stopping acquisition: {ex}")

#     def _capture_live_frame(self):
#         try:
#             image_result = self.cam.GetNextImage(1000)
#             if not image_result.IsIncomplete():
#                 image_data = image_result.GetNDArray()
#                 self.new_live_frame.emit(image_data)
#             image_result.Release()
#         except SpinnakerException:
#             self.stop_live_view()
#             self.connection_lost.emit("Camera disconnected during live view.")

#     def start_acquisition(self, num_images: int, save_dir: str):
#         if not self.is_connected or self.is_acquiring:
#             self.status_update.emit("Cannot start acquisition: Not connected or already busy.")
#             return
#         try:
#             self.is_acquiring = True
#             self.cam.BeginAcquisition()
#             self.status_update.emit(f"Starting acquisition of {num_images} images...")
#             self.acquisition_thread = AcquisitionThread(self.cam, num_images, save_dir, self)
#             self.acquisition_thread.finished.connect(self._on_acquisition_finished)
#             self.acquisition_thread.progress.connect(self.acquisition_progress)
#             self.acquisition_thread.error.connect(lambda msg: self.status_update.emit(msg))
#             self.acquisition_thread.start()
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Acquisition Error: {ex}")
#             self.is_acquiring = False

#     def _on_acquisition_finished(self, save_dir: str):
#         try:
#             self.cam.EndAcquisition()
#         except SpinnakerException as ex:
#             self.status_update.emit(f"Error ending acquisition: {ex}")
#         self.is_acquiring = False
#         self.acquisition_finished.emit(save_dir)

#     def start_aggregation(self, threshold: int = 20):
#         """
#         Starts a continuous, real-time aggregation process.
#         This will run until stop_aggregation() is called.
        
#         Args:
#             threshold (int): The brightness threshold to ignore noise.
#         """
#         if not self.is_connected or self.is_acquiring or self.is_aggregating:
#             self.status_update.emit("Error: Cannot start aggregation. Controller is busy or not connected.")
#             return

#         self.status_update.emit("Starting continuous aggregation... Press 's' in the display window to stop.")
#         self.is_aggregating = True
        
#         self.aggregation_threshold = threshold
#         self.aggregated_image = None  # Reset the canvas

#         try:
#             self.cam.BeginAcquisition()
#             self.new_live_frame.connect(self._aggregate_frame)
            
#             # Start the timer to poll for frames
#             self._live_view_timer.setInterval(1) 
#             # --- THE FIX: Use the correct variable name with the underscore ---
#             self._live_view_timer.start()
#             # --- END OF FIX ---

#         except SpinnakerException as ex:
#             self.status_update.emit(f"Aggregation Start Error: {ex}")
#             self.is_aggregating = False

#     def stop_aggregation(self):
#         """Manually stops an ongoing aggregation process and finalizes the result."""
#         if not self.is_aggregating:
#             return

#         self.is_aggregating = False
        
#         # --- FIX: Stop the timer here as well ---
#         self._live_view_timer.stop()
#         # --- END OF FIX ---
        
#         try:
#             # Disconnect the signal so it doesn't keep trying to aggregate
#             self.new_live_frame.disconnect(self._aggregate_frame)
#             self.cam.EndAcquisition()
#             self.status_update.emit("Aggregation stopped.")
            
#             if self.aggregated_image is not None:
#                 self.aggregation_finished.emit(self.aggregated_image)

#         except SpinnakerException as ex:
#             self.status_update.emit(f"Error stopping aggregation: {ex}")
#         except TypeError: 
#             pass

#     def _initialize_aggregation(self, first_frame: np.ndarray):
#         """Internal method to set up the aggregation canvas."""
#         self.status_update.emit("First frame received. Initializing aggregation canvas.")
#         # Create a blank (black) canvas of the same size and type
#         self.aggregated_image = np.zeros_like(first_frame, dtype=np.uint8)

#     def _aggregate_frame(self, new_frame: np.ndarray):
#         """
#         This method is connected as a slot to the new_live_frame signal
#         during aggregation mode. It now runs indefinitely.
#         """
#         if not self.is_aggregating:
#             return

#         if self.aggregated_image is None:
#             self._initialize_aggregation(new_frame)

#         # --- Core aggregation logic (unchanged) ---
#         if len(new_frame.shape) == 3:
#             gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = new_frame
#         processed_frame = np.where(gray > self.aggregation_threshold, gray, 0).astype(np.uint8)
#         self.aggregated_image = np.maximum(self.aggregated_image, processed_frame)

#         self.aggregation_updated.emit(self.aggregated_image.copy())

#     # --- You also need to modify the _capture_live_frame method ---
#     # to ensure it emits the signal that _aggregate_frame is listening to.
    
#     def _capture_live_frame(self):
#         """
#         Grabs a single frame and emits it. This now serves both live view
#         and aggregation modes.
#         """
#         try:
#             image_result = self.cam.GetNextImage(1000)
#             if not image_result.IsIncomplete():
#                 # Convert to a NumPy array that can be shared
#                 image_data = image_result.GetNDArray().copy()
#                 self.new_live_frame.emit(image_data)
#             image_result.Release()
#         except SpinnakerException:
#             # If something goes wrong, stop everything
#             if self.is_aggregating:
#                 self.stop_aggregation()
#             if self.is_acquiring: # This is the live view flag
#                 self.stop_live_view()
#             self.connection_lost.emit("Camera disconnected during capture.")

# File: camera_controller.py

"""
Module for controlling a FLIR (formerly Point Grey) camera using the PySpin SDK.

This module provides the CameraController class, which abstracts the SDK calls
for camera initialization, configuration, live view, and threaded bulk acquisition.
"""
import os
import sys
import numpy as np
from PySpin.PySpin import System, CameraPtr, SpinnakerException, CStringPtr, CEnumerationPtr, IsReadable, IsWritable, CFloatPtr
import cv2

from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer


class AcquisitionThread(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    error = pyqtSignal(str)

    def __init__(self, cam: CameraPtr, num_images: int, save_dir: str, parent=None):
        super().__init__(parent)
        self.cam = cam
        self.num_images = num_images
        self.save_dir = save_dir

    def run(self):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            nodemap_tldevice = self.cam.GetTLDeviceNodeMap()
            node_device_serial_number = CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            device_serial_number = node_device_serial_number.GetValue() if IsReadable(node_device_serial_number) else ''
            
            for i in range(self.num_images):
                image_result = self.cam.GetNextImage(1000)
                if image_result.IsIncomplete():
                    self.error.emit(f'Image {i} incomplete with status {image_result.GetImageStatus()}')
                else:
                    filename = f'Acquisition-{device_serial_number}-{i}.jpg' if device_serial_number else f'Acquisition-{i}.jpg'
                    full_path = os.path.join(self.save_dir, filename)
                    image_result.Save(full_path)
                    self.progress.emit(i + 1, self.num_images)
                image_result.Release()
            self.finished.emit(self.save_dir)
        except SpinnakerException as ex:
            self.error.emit(f"Acquisition Error: {ex}")


class CameraController(QObject):
    # --- Signals ---
    status_update = pyqtSignal(str)
    connection_lost = pyqtSignal(str)
    new_live_frame = pyqtSignal(object) # Emits raw frame for live view
    acquisition_finished = pyqtSignal(str)
    acquisition_progress = pyqtSignal(int, int)
    aggregation_finished = pyqtSignal(object) # Emits the final aggregated NumPy array
    
    # *** NEW SIGNAL FOR REAL-TIME AGGREGATION VIEW ***
    aggregation_updated = pyqtSignal(object) # Emits the in-progress aggregated image
    exposure_time_updated = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.system = None
        self.cam = None
        self.cam_list = None
        self.nodemap = None
        self.is_connected = False
        self.is_acquiring = False # Flag for live view or bulk acquisition
        self.is_aggregating = False # Specific flag for aggregation mode
        self.acquisition_thread = None
        
        self._live_view_timer = QTimer(self)
        self._live_view_timer.timeout.connect(self._capture_frame_for_processing)

        self.aggregated_image = None
        self.aggregation_threshold = 20

    def connect(self):
        if self.is_connected: return True
        try:
            self.system = System.GetInstance()
            self.cam_list = self.system.GetCameras()

            if self.cam_list.GetSize() == 0:
                self.status_update.emit("Error: No cameras detected.")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                self.system = None
                return False

            self.cam = self.cam_list.GetByIndex(0)
            self.cam.Init()
            self.nodemap = self.cam.GetNodeMap()
            if not self._configure_camera():
                self.cam.DeInit()
                del self.cam
                return False
            self.is_connected = True
            self.status_update.emit("Camera connected and configured successfully.")
            return True
        except SpinnakerException as ex:
            self.status_update.emit(f"Camera Connection Error: {ex}")
            return False

    def disconnect(self):
        if not self.is_connected: return
        if self.is_acquiring or self.is_aggregating:
             self.stop_live_view()
             self.stop_aggregation()

        if self.cam:
            try:
                if self.cam.IsInitialized():
                    self.cam.DeInit()
            except SpinnakerException as ex:
                self.status_update.emit(f"Error during DeInit: {ex}")
            del self.cam
            self.cam = None
        
        if self.cam_list:
            try:
                self.cam_list.Clear()
            except SpinnakerException as ex:
                self.status_update.emit(f"Error clearing camera list: {ex}")
            self.cam_list = None
        
        if self.system:
            try:
                self.system.ReleaseInstance()
            except SpinnakerException as ex:
                self.status_update.emit(f"Error releasing system instance: {ex}")
            self.system = None
        
        self.is_connected = False
        self.status_update.emit("Camera disconnected.")

    def _configure_camera(self):
        try:
            node_acq_mode = CEnumerationPtr(self.nodemap.GetNode('AcquisitionMode'))
            node_acq_mode_continuous = node_acq_mode.GetEntryByName('Continuous')
            node_acq_mode.SetIntValue(node_acq_mode_continuous.GetValue())
            self.status_update.emit("Acquisition mode set to Continuous.")
            
            node_pixel_format = CEnumerationPtr(self.nodemap.GetNode('PixelFormat'))
            node_pixel_format_mono8 = node_pixel_format.GetEntryByName('Mono8')
            node_pixel_format.SetIntValue(node_pixel_format_mono8.GetValue())
            self.status_update.emit("Pixel format set to Mono8.")
            return True
        except SpinnakerException as ex:
            self.status_update.emit(f"Configuration Error: {ex}")
            return False

    def get_exposure_limits(self):
        """Queries the camera for its min and max exposure time in microseconds."""
        if not self.is_connected or not self.nodemap:
            return None, None
        try:
            node_exposure_time = CFloatPtr(self.nodemap.GetNode('ExposureTime'))
            if not IsReadable(node_exposure_time):
                self.status_update.emit("Error: ExposureTime node is not readable.")
                return None, None
            return node_exposure_time.GetMin(), node_exposure_time.GetMax()
        except SpinnakerException as ex:
            self.status_update.emit(f"Error getting exposure limits: {ex}")
            return None, None
        
    # ### MODIFIED ### Updated to emit the new signal
    def set_exposure_time(self, exposure_us: float):
        if not self.is_connected:
            self.status_update.emit("Error: Cannot set exposure. Camera not connected.")
            return False
        try:
            # Ensure auto-exposure is off
            node_exposure_auto = CEnumerationPtr(self.nodemap.GetNode('ExposureAuto'))
            if IsWritable(node_exposure_auto):
                entry_exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
                node_exposure_auto.SetIntValue(entry_exposure_auto_off.GetValue())
            
            node_exposure_time = CFloatPtr(self.nodemap.GetNode('ExposureTime'))
            if IsWritable(node_exposure_time):
                min_val, max_val = node_exposure_time.GetMin(), node_exposure_time.GetMax()
                
                # Clamp the value to the hardware limits
                value_to_set = min(max(exposure_us, min_val), max_val)
                
                node_exposure_time.SetValue(value_to_set)
                actual_value = node_exposure_time.GetValue() # Read back the actual value

                self.status_update.emit(f"Exposure time set to {actual_value:.2f} µs.")
                self.exposure_time_updated.emit(actual_value) # Emit signal with actual value
                return True
            else:
                self.status_update.emit("Error: Exposure time node is not writable.")
                return False
        except SpinnakerException as ex:
            self.status_update.emit(f"Error setting exposure: {ex}")
            return False

    def start_live_view(self, interval_ms: int = 33):
        if not self.is_connected or self.is_acquiring or self.is_aggregating: return
        try:
            self.is_acquiring = True
            self.cam.BeginAcquisition()
            self._live_view_timer.setInterval(interval_ms)
            self._live_view_timer.start()
            self.status_update.emit("Live view started.")
        except SpinnakerException as ex:
            self.status_update.emit(f"Live View Error: {ex}")
            self.is_acquiring = False

    def stop_live_view(self):
        self._live_view_timer.stop()
        if self.is_acquiring:
            try:
                # Check if acquisition is running before trying to end it
                if self.cam.IsStreaming():
                    self.cam.EndAcquisition()
                self.status_update.emit("Live view stopped.")
            except SpinnakerException as ex:
                self.status_update.emit(f"Error stopping acquisition: {ex}")
            finally:
                 self.is_acquiring = False

    def _capture_frame_for_processing(self):
        """
        Grabs a single frame and emits it. This now serves both live view
        and aggregation modes by routing the frame to the correct handler.
        """
        try:
            image_result = self.cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                image_data = image_result.GetNDArray().copy()
                
                # Route the frame based on the current mode
                if self.is_aggregating:
                    self._aggregate_frame(image_data)
                elif self.is_acquiring:
                    self.new_live_frame.emit(image_data)

            image_result.Release()
        except SpinnakerException:
            self.connection_lost.emit("Camera disconnected during capture.")
            self.stop_live_view()
            self.stop_aggregation()

    def start_acquisition(self, num_images: int, save_dir: str):
        if not self.is_connected or self.is_acquiring or self.is_aggregating:
            self.status_update.emit("Cannot start acquisition: Not connected or already busy.")
            return
        try:
            self.is_acquiring = True
            self.cam.BeginAcquisition()
            self.status_update.emit(f"Starting acquisition of {num_images} images...")
            self.acquisition_thread = AcquisitionThread(self.cam, num_images, save_dir, self)
            self.acquisition_thread.finished.connect(self._on_acquisition_finished)
            self.acquisition_thread.progress.connect(self.acquisition_progress)
            self.acquisition_thread.error.connect(lambda msg: self.status_update.emit(msg))
            self.acquisition_thread.start()
        except SpinnakerException as ex:
            self.status_update.emit(f"Acquisition Error: {ex}")
            self.is_acquiring = False

    def _on_acquisition_finished(self, save_dir: str):
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
        except SpinnakerException as ex:
            self.status_update.emit(f"Error ending acquisition: {ex}")
        self.is_acquiring = False
        self.acquisition_finished.emit(save_dir)

    def start_aggregation(self, threshold: int = 20):
        if not self.is_connected or self.is_acquiring or self.is_aggregating:
            self.status_update.emit("Error: Cannot start aggregation. Controller is busy.")
            return

        self.is_aggregating = True
        self.aggregation_threshold = threshold
        self.aggregated_image = None  # Reset the canvas
        self.status_update.emit(f"Starting continuous aggregation with threshold {threshold}.")

        try:
            self.cam.BeginAcquisition()
            self._live_view_timer.setInterval(1) # As fast as possible
            self._live_view_timer.start()
        except SpinnakerException as ex:
            self.status_update.emit(f"Aggregation Start Error: {ex}")
            self.is_aggregating = False

    def stop_aggregation(self):
        if not self.is_aggregating: return
        
        self.is_aggregating = False
        self._live_view_timer.stop()
        
        try:
            if self.cam.IsStreaming():
                self.cam.EndAcquisition()
            self.status_update.emit("Aggregation stopped.")
            
            if self.aggregated_image is not None:
                self.aggregation_finished.emit(self.aggregated_image)
        except SpinnakerException as ex:
            self.status_update.emit(f"Error stopping aggregation: {ex}")

    def _initialize_aggregation(self, first_frame: np.ndarray):
        self.status_update.emit("First frame received. Initializing aggregation canvas.")
        self.aggregated_image = np.zeros_like(first_frame, dtype=np.uint8)

    def _aggregate_frame(self, new_frame: np.ndarray):
        if not self.is_aggregating: return

        if self.aggregated_image is None:
            self._initialize_aggregation(new_frame)

        # Core aggregation logic
        if len(new_frame.shape) == 3: # If somehow a color image is passed
            gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = new_frame
            
        processed_frame = np.where(gray > self.aggregation_threshold, gray, 0).astype(np.uint8)
        self.aggregated_image = np.maximum(self.aggregated_image, processed_frame)

        # *** EMIT THE UPDATE SIGNAL ***
        # This sends the current state of the aggregated image to the GUI
        self.aggregation_updated.emit(self.aggregated_image)