# setup.py

import sys
import os
from cx_Freeze import setup, Executable

# --- Automatically determine the project root and venv path ---
# This makes the script robust, even with an external venv.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENV_DIR = os.path.abspath(os.path.join(_PROJECT_DIR, 'build_venv'))

# --- Find and validate the PySpin SDK path ---
# This is the most critical part for your project.
PYSPIN_PATH = os.path.join(_VENV_DIR, 'Lib', 'site-packages', 'PySpin')
# On Linux/macOS, the path might be 'lib/pythonX.Y/site-packages/...'
if not os.path.isdir(PYSPIN_PATH):
    raise FileNotFoundError(f"PySpin SDK path not found. Please verify the path is correct: {PYSPIN_PATH}")

# --- List all files and directories to be included ---
include_files = [
    (os.path.join(PYSPIN_PATH, ''), 'lib/PySpin'),
    ('assets', 'assets'),
    ('calibration_wizard', 'calibration_wizard'),
    ('dedicated_camera_app.py', 'dedicated_camera_app.py'),
    ('core', 'core'),
    ('hardware', 'hardware'),
]

# ### MODIFIED: Added function to handle numpy DLLs ###
def get_numpy_include_files():
    """
    Finds and returns the path to numpy's core DLLs.
    This is a robust way to handle the 'numpy.core._methods' error.
    """
    numpy_base_path = os.path.join(_VENV_DIR, 'Lib', 'site-packages', 'numpy', 'core')
    files_to_include = []
    for filename in os.listdir(numpy_base_path):
        if filename.endswith(('.dll', '.pyd')):
            source_path = os.path.join(numpy_base_path, filename)
            # The destination is 'lib/numpy/core' to match the package structure
            dest_path = os.path.join('lib', 'numpy', 'core', filename)
            files_to_include.append((source_path, dest_path))
    return files_to_include

# Add the numpy files to our main include list
include_files.extend(get_numpy_include_files())


# --- Build options ---
# ### MODIFIED: This section is updated for better dependency detection ###
build_exe_options = {
    # We now explicitly include scipy, as it's a dependency of spectral-python
    "packages": [
        "os", "sys", "json", "traceback", "subprocess", "time", "datetime",
        "PyQt6", "cv2", "numpy", "h5py", "serial", "spectral", "scipy",
        "unittest"
    ],
    "includes": [
        "spectral.io.envi",
    ],
    "include_files": include_files,
    "excludes": ["tkinter"],
    "optimize": 2,
}

# --- MSI Installer options (for Windows) ---
bdist_msi_options = {
    'add_to_path': False,
    'initial_target_dir': r'%ProgramFiles%\HSISuite',
    'upgrade_code': '{c0a8767e-4d42-4985-98a7-a62555e1c406}' # Replace with your own GUID
}

# --- Define the executable ---
base = None
if sys.platform == "win32":
    base = "Win32GUI"

main_executable = Executable(
    "hsi_control_v5.py",
    base=base,
    target_name="HSI_Control_App.exe",
    icon="assets/256_icon.ico"
)

# --- Run the setup ---
setup(
    name="HSI Control Suite",
    version="1.0",
    description="Hyperspectral Imaging Control and Acquisition Software",
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,
    },
    executables=[main_executable]
)