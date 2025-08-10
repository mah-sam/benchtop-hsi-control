# HSI Control Suite: An Open-Source GUI for Hyperspectral Imaging

<div align="center">
  <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/hsi_banner.png" alt="HSI Control Suite Banner" width="800"/>
</div>

<p align="center">
  <strong>Unified Control, Calibration, and Data Acquisition for Benchtop Push-Broom Hyperspectral Imaging Systems</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Framework-PyQt6-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

---

## 1. Introduction

The **HSI Control Suite** is a complete, open-source software application designed to manage the entire data acquisition pipeline for custom-built, benchtop push-broom hyperspectral imaging (HSI) systems. Developed in Python with a professional PyQt6 graphical user interface (GUI), this suite addresses the significant challenges of cost, operational complexity, and fragmented software workflows that often hinder HSI research.

By integrating hardware control, a guided multi-step calibration wizard, automated scan execution, and powerful post-processing tools into a single, unified platform, this software transforms a collection of disparate hardware components into a cohesive, user-friendly scientific instrument.

The primary outcome is a fully validated acquisition platform that streamlines the workflow from hardware setup to the generation of analysis-ready data packages, enabling researchers to create high-quality, reproducible hyperspectral datasets with ease.

## 2. The Problem: A Fragmented Workflow

Hyperspectral imaging is a powerful technique, but building and operating a custom system is notoriously complex. Researchers often face a fragmented and inefficient workflow, relying on a patchwork of disconnected tools:
- **Manufacturer SDKs:** Low-level, command-line tools for basic camera control.
- **Custom Scripts:** Separate Python or MATLAB scripts to control motorized stages.
- **Manual Synchronization:** Error-prone manual coordination between camera capture and stage movement.
- **Separate Calibration Software:** Complex, offline tools for geometric and spectral correction.
- **Post-Processing Hassles:** Manually assembling data cubes and adding metadata in yet another software environment.

This fragmentation creates a steep learning curve, introduces opportunities for error, and hinders the creation of consistent, high-quality datasets.

<div align="center">
  <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/workflow_comparison.png" alt="Workflow Comparison Diagram" width="900"/>
  <br>
  <em>Figure 1: Conceptual comparison between the common fragmented HSI workflow and the integrated, streamlined workflow enabled by the HSI Control Suite.</em>
</div>

## 3. Our Solution: Key Features

The HSI Control Suite is designed from the ground up to solve these problems by providing a seamless, end-to-end solution.

- **Unified Graphical User Interface (GUI):** A professional and intuitive interface built with PyQt6 provides a central control panel for all system operations, eliminating the need for command-line interaction.

- **Integrated Hardware Control:** Seamless, multi-threaded control of both the hyperspectral camera (via Spinnaker SDK) and the linear motion stage (via Arduino/Serial communication).

- **Guided Calibration Wizard:** A step-by-step wizard that guides the user through the entire geometric and spectral calibration process, generating a master calibration file that ensures data accuracy and consistency.

- **Automated & Synchronized Acquisition:** A dedicated acquisition dialog automates the entire push-broom scan. It precisely synchronizes camera frame capture with stage motion based on user-defined parameters (speed, duration, FPS), ensuring geometrically correct data cubes.

- **Real-Time Corrected Live View:** A critical feature for push-broom systems. The live camera feed is geometrically corrected in real-time using the calibration data, allowing for accurate sample positioning and focusing.

- **Robust HDF5 Data Management:** Scans are saved as single, self-describing HDF5 files. Each file contains:
    - The full hyperspectral data cube.
    - A complete set of acquisition metadata (camera settings, scan parameters, etc.).
    - User-defined labels for sample tracking and ground-truth data.
    - An automatically generated RGB preview for quick qualitative assessment.

- **Integrated Post-Processing & Analysis Tools:**
    - **Post-Scan Labeling:** Immediately add or edit metadata labels in the saved HDF5 file without needing external tools.
    - **Interactive Cropping:** Define a spatial region of interest on the scan preview and save a new, smaller data cube.
    - **Advanced Slice Analyzer:** An interactive dialog to explore the data cube slice-by-slice, plot spectra from any pixel, and perform robust Region of Interest (ROI) analysis using "Magic Wand" and brush tools.

- **Open-Source and Extensible:** The entire codebase is written in Python, making it easy to modify, extend, or integrate with other scientific libraries.

## 4. System Architecture

The software is built on a modular, multi-threaded architecture to ensure a responsive user experience and reliable hardware communication. The GUI interacts with a hardware abstraction layer, which manages the low-level communication with the camera and stage on separate threads.

<div align="center">
  <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/system_architecture.png" alt="System Architecture Diagram" width="700"/>
  <br>
  <em>Figure 2: High-level software architecture of the HSI Control Suite.</em>
</div>

## 5. Gallery

<table align="center">
  <tr>
    <td align="center">
      <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/main_gui.png" alt="Main GUI Screenshot" width="450">
      <br><em>Figure 3: The main control panel, showing the real-time corrected live view and hardware control modules.</em>
    </td>
    <td align="center">
      <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/acquisition_dialog.png" alt="Acquisition Dialog Screenshot" width="450">
      <br><em>Figure 4: The automated acquisition dialog with advanced save options and a live RGB preview of the ongoing scan.</em>
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="https://raw.githubusercontent.com/MahmoudSameh/HSI-Control-Suite/main/docs/images/slice_analyzer.png" alt="Slice Analyzer Screenshot" width="800">
      <br><em>Figure 5: The Advanced Slice Analyzer, showing an interactive spectral slice with a user-defined Region of Interest (ROI) and the resulting averaged spectrum.</em>
    </td>
  </tr>
</table>

## 6. Hardware Requirements

This software is designed to control a specific set of COTS and custom-fabricated hardware. While adaptable, the default configuration requires:

- **Camera:** A FLIR (formerly Point Grey) machine vision camera compatible with the **Spinnaker SDK**. (Tested with FLIR Grasshopper3).
- **Linear Stage:** A stepper motor-driven linear stage.
- **Stage Controller:** An **Arduino Uno** (or compatible board) running custom firmware to control a stepper motor driver (e.g., A4988, DRV8825).
- **Motor:** A **NEMA-17** stepper motor.
- **Illumination:** Stable, broad-spectrum lighting (e.g., halogen lamps).

## 7. Installation

You can run the HSI Control Suite either from a pre-built executable or by running the source code directly.

### A. From Executable (Recommended for End-Users)

1.  Navigate to the [**Releases**](https://github.com/MahmoudSameh/HSI-Control-Suite/releases) page of this repository.
2.  Download the latest installer (`HSI_Control_App-vX.X.X.msi`) or the zipped executable package.
3.  Run the installer or extract the zip file.
4.  Launch the `HSI_Control_App.exe`.

### B. From Source (Recommended for Developers)

**Prerequisites:**
- Python 3.10 or newer.
- **Crucial:** You must install the **FLIR Spinnaker SDK** from the official FLIR website. Make sure to install the Python bindings (`PySpin`) during the SDK installation.

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MahmoudSameh/HSI-Control-Suite.git
    cd HSI-Control-Suite
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python hsi_control_v5.py
    ```

## 8. Usage Workflow

The software is designed to guide the user through a logical workflow:

1.  **Connect Hardware:** Use the "Connect All Devices" button in the main GUI to establish communication with the camera and stage controller.
2.  **Perform Calibration (First-Time Use):**
    - Go to `Calibration > Run Calibration Wizard...`.
    - Follow the on-screen instructions for each step (Straightening, Cropping, Spectral Calibration).
    - The wizard will generate a `master_calibration.json` file in the `assets` directory. The main application will load this automatically on startup.
3.  **Set Scan Parameters:**
    - Manually move the stage to the desired start and end positions and click "Set Current as Start" and "Set Current as End".
    - Configure the scan speed and other parameters in the "Scan Configuration" panel.
4.  **Acquire Data Cube:**
    - Click "Acquire Data Cube..." to open the acquisition dialog.
    - Fill in any necessary metadata labels.
    - Choose a save location and filename.
    - Click "Start Scan" to begin the automated acquisition process.
5.  **Analyze and Process:**
    - After the scan, the RGB preview will appear in the main window.
    - Use the "Post-Scan Labeling" and "Post-Scan Cropping" tools as needed.
    - For in-depth analysis, launch the "Advanced Slice Analyzer" from the `Tools` menu and open the newly created HDF5 file.

## 9. Output Data Format: HDF5 Structure

The HSI Control Suite produces a single, comprehensive HDF5 file for each scan, ensuring data integrity and portability. The internal structure is as follows:

```
/ (Root Group)
├── Attributes:
│   ├── 'metadata': (String) A JSON-formatted string containing all acquisition parameters,
│   │               calibration info, timestamps, etc.
│   ├── 'labels': (String) A JSON-formatted string with user-defined key-value pairs.
│   └── 'roi_settings': (String, Optional) JSON string with parameters used to generate a saved ROI.
│
├── Datasets:
│   ├── 'cube': (Dataset, float32) The main hyperspectral data cube.
│   │           Shape: (spectral_height, spatial_width, num_bands)
│   │           Dimensions: (Spectral Axis, Spatial Axis, Scan Axis)
│   │
│   ├── 'rgb_preview': (Dataset, uint8) The 3-channel RGB preview image.
│   │                  Shape: (scan_length_pixels, spatial_width_pixels, 3)
│   │
│   └── 'roi_mask': (Dataset, bool, Optional) A 2D boolean mask defining a saved ROI.
│                   Shape: (scan_length_pixels, spatial_width_pixels)
```

## 10. Contributing

Contributions are welcome! If you would like to contribute to the project, please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please report any bugs or suggest features by opening an issue on the GitHub repository.

## 11. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 12. Acknowledgments & Citation

This research was supported by the Undergraduate Research Office and Electrical Engineering Department at King Fahd University of Petroleum and Minerals (KFUPM) through the KFUPM Inbound Summer Research Program (T243).

We thank Ibrahim Azeem for assistance with microcontroller firmware and Asim Al-Qarni for support during system fabrication.

If you use this software in your research, please cite it as follows:

```bibtex
@software{Sameh_HSI_Control_Suite_2025,
  author = {Sameh, Mahmoud and Albeladi, Ali},
  title = {{HSI Control Suite: An Open-Source GUI for Real-Time Control and Data Acquisition in Benchtop Push-Broom Hyperspectral Imaging Systems}},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/MahmoudSameh/HSI-Control-Suite}
}
```
