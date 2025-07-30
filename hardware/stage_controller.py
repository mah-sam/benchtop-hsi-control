"""
Module for controlling the Arduino-based stepper motor stage.

This module provides the StageController class, which abstracts the serial
communication protocol for the stepper motor stage, making it easy to integrate
into a PyQt application.
"""

import struct
import time
import serial
import serial.tools.list_ports

from PyQt6.QtCore import QObject, pyqtSignal, QThread


class SerialReaderThread(QThread):
    """
    A dedicated thread to continuously read data from the serial port
    without blocking the main application GUI.
    """
    data_received = pyqtSignal(str)

    def __init__(self, serial_port, parent=None):
        """
        Initializes the reader thread.
        Args:
            serial_port (serial.Serial): The active serial port instance.
            parent (QObject): The parent object.
        """
        super().__init__(parent)
        self.ser = serial_port
        self.running = True

    def run(self):
        """Main execution loop for the thread."""
        while self.running:
            if self.ser and self.ser.is_open and self.ser.in_waiting:
                try:
                    # Read a line, decode it, and strip whitespace
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.data_received.emit(line)
                except serial.SerialException:
                    # Port might have been disconnected
                    self.running = False
            # A small sleep to prevent this loop from running too hot
            time.sleep(0.05)

    def stop(self):
        """Stops the thread gracefully."""
        self.running = False
        self.wait()


class StageController(QObject):
    """
    Manages all communication with the Arduino-controlled stepper motor stage.

    This class handles port detection, connection, command sending, and
    parsing feedback from the Arduino. It uses PyQt signals to communicate
    its state to the rest of the application.
    """
    # --- Signals for GUI communication ---
    status_update = pyqtSignal(str)      # For general log messages
    homing_complete = pyqtSignal()       # Emitted when the system is ready
    connection_lost = pyqtSignal()       # Emitted if the serial connection fails

    def __init__(self, parent=None):
        """Initializes the StageController."""
        super().__init__(parent)
        self.serial = None
        self.reader_thread = None
        self.is_connected = False
        self.is_homed = False

    def _find_arduino_port(self):
        """Scans COM ports to find a connected Arduino."""
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            desc = port.description.lower()
            # Common keywords for Arduino or clones
            if "arduino" in desc or "ch340" in desc or "usb serial" in desc:
                return port.device
        return None

    def connect(self):
        """
        Attempts to find and connect to the Arduino stage.
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        if self.is_connected:
            self.status_update.emit("Already connected.")
            return True

        port = self._find_arduino_port()
        if not port:
            self.status_update.emit("Error: Arduino controller not found.")
            return False

        try:
            self.serial = serial.Serial(port, 9600, timeout=0.1)
            # Wait for the Arduino to reset after establishing connection
            time.sleep(2)
            self.is_connected = True
            self.status_update.emit(f"Successfully connected to Arduino on {port}.")
            self._start_reader_thread()
            return True
        except serial.SerialException as e:
            self.status_update.emit(f"Error: Failed to open port {port}. {e}")
            return False

    def disconnect(self):
        """Closes the serial connection and stops the reader thread."""
        if self.reader_thread:
            self.reader_thread.stop()
            self.reader_thread = None

        if self.serial and self.serial.is_open:
            self.serial.close()
            self.serial = None

        self.is_connected = False
        self.is_homed = False
        self.status_update.emit("Disconnected from stage controller.")

    def _start_reader_thread(self):
        """Initializes and starts the background thread for serial reading."""
        if self.reader_thread:
            self.reader_thread.stop()

        self.reader_thread = SerialReaderThread(self.serial, self)
        self.reader_thread.data_received.connect(self._handle_serial_data)
        self.reader_thread.start()
        self.status_update.emit("Started listening for Arduino messages.")

    def _handle_serial_data(self, data: str):
        """
        Parses messages received from the Arduino and emits appropriate signals.
        This method acts as a slot for the reader thread's signal.
        """
        self.status_update.emit(f"Arduino: {data}")

        if "System ready" in data:
            self.is_homed = True
            self.homing_complete.emit()

    def move_to(self, position: int, speed: int = 100):
        """
        Sends a command to move the stage to a specific position at a given speed.

        Args:
            position (int): The target position (valid range: 10-250).
            speed (int): The target speed (valid range: 50-1000).
        """
        if not self.is_connected:
            self.status_update.emit("Error: Not connected to stage. Cannot send command.")
            return

        if not self.is_homed:
            self.status_update.emit("Error: Stage not homed. Cannot send move command.")
            return

        # --- Input Validation ---
        if not (10 <= position <= 250):
            self.status_update.emit(f"Warning: Position {position} is outside valid range (10-250).")
            return
        if not (50 <= speed <= 1000):
            self.status_update.emit(f"Warning: Speed {speed} is outside valid range (50-1000).")
            return

        try:
            # --- Pack data into a 3-byte binary packet ---
            # '<' for little-endian byte order
            # 'B' for unsigned char (1 byte) for position
            # 'H' for unsigned short (2 bytes) for speed
            command_packet = struct.pack('<BH', position, speed)
            self.serial.write(command_packet)
            self.status_update.emit(f"Command sent: Move to {position} at speed {speed}.")
        except (serial.SerialException, AttributeError) as e:
            self.status_update.emit(f"Error sending command: {e}")
            self.connection_lost.emit()