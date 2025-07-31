# In core/file_io.py

import os
import json
import sys
import h5py
import numpy as np
from spectral import envi, SpyFile
from typing import Dict, Any, Tuple, Optional

# --- ENVI Format Functions (Unchanged) ---
def save_envi(filepath: str, data_cube: np.ndarray, metadata: dict):
    if not all(k in metadata for k in ['wavelength', 'interleave']):
        raise ValueError("Metadata must contain 'wavelength' and 'interleave' keys.")
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    envi.save_image(filepath, data_cube, metadata=metadata, force=True)
    print(f"ENVI file saved successfully to {filepath}")

def load_envi(filepath: str) -> tuple[np.ndarray, dict]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENVI header file not found at: {filepath}")
    img: SpyFile = envi.open(filepath)
    data_cube = img.load()
    metadata = dict(img.metadata)
    if 'wavelength' in metadata and isinstance(metadata['wavelength'], str):
        metadata['wavelength'] = [float(w) for w in metadata['wavelength'].strip('{} \n').split(',')]
    print(f"ENVI file loaded successfully from {filepath}")
    return data_cube, metadata

# --- HDF5 Format Functions (Modified) ---

def save_h5(
    filepath: str,
    data_cube: np.ndarray,
    metadata: Dict[str, Any],
    labels: Optional[Dict[str, Any]] = None,
    rgb_preview: Optional[np.ndarray] = None,
    compression_algo: str = 'blosc',
    compression_level: int = 9
):
    """
    Saves hyperspectral data to HDF5 with user-configurable compression.
    Uses a try-except block to robustly fall back to gzip if the preferred
    compressor is unavailable.
    """
    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    final_algo = compression_algo

    with h5py.File(filepath, 'w') as f:
        try:
            f.create_dataset('cube', data=data_cube, compression=compression_algo, compression_opts=compression_level)
        except ValueError as e:
            if "is unavailable" in str(e):
                print(f"WARNING: Compression filter '{compression_algo}' not available.")
                print(f"         Falling back to 'gzip' for this save operation.")
                print(f"         To enable '{compression_algo}', you may need to run: pip install {compression_algo} h5py --force-reinstall")
                
                f.create_dataset('cube', data=data_cube, compression='gzip', compression_opts=compression_level)
                final_algo = 'gzip' 
            else:
                raise e

        if rgb_preview is not None:
            f.create_dataset('rgb_preview', data=rgb_preview, compression=final_algo, compression_opts=compression_level)

        f.attrs['metadata'] = json.dumps(metadata)
        if labels:
            f.attrs['labels'] = json.dumps(labels)
    
    print(f"HDF5 file saved successfully to {filepath} using '{final_algo}' compression.")

def load_h5(filepath: str) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]:
    """
    Loads a hyperspectral data cube and all associated data from an HDF5 file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found at: {filepath}")

    with h5py.File(filepath, 'r') as f:
        if 'cube' not in f:
            raise KeyError("HDF5 file is missing the 'cube' dataset.")
            
        data_cube = f['cube'][:]
        metadata_str = f.attrs.get('metadata', '{}')
        metadata = json.loads(metadata_str)
        labels_str = f.attrs.get('labels', '{}')
        labels = json.loads(labels_str)
        
        rgb_preview = None
        if 'rgb_preview' in f:
            rgb_preview = f['rgb_preview'][:]
    
    print(f"HDF5 file loaded successfully from {filepath}")
    return data_cube, metadata, labels, rgb_preview

# ### NEW FUNCTION ###
def update_h5_labels(filepath: str, labels: Dict[str, Any]):
    """
    Efficiently updates only the 'labels' attribute of an existing HDF5 file
    without reading or rewriting the entire data cube.

    Args:
        filepath (str): The path to the HDF5 file to update.
        labels (Dict[str, Any]): The new dictionary of labels to save.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found at: {filepath}")

    try:
        # Open the file in read/write mode ('r+')
        with h5py.File(filepath, 'r+') as f:
            # Overwrite the 'labels' attribute directly.
            f.attrs['labels'] = json.dumps(labels)
        print(f"Successfully updated labels in {filepath}")
    except Exception as e:
        print(f"Error updating labels in {filepath}: {e}")
        raise

def load_h5_preview_and_metadata(filepath: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[np.ndarray]]:
    """
    Efficiently loads only the metadata, labels, and RGB preview from an HDF5
    file without loading the main (and potentially very large) data cube.
    This is ideal for fast previews and metadata inspection.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        A tuple containing (metadata_dict, labels_dict, rgb_preview_array).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found at: {filepath}")

    with h5py.File(filepath, 'r') as f:
        metadata_str = f.attrs.get('metadata', '{}')
        metadata = json.loads(metadata_str)
        labels_str = f.attrs.get('labels', '{}')
        labels = json.loads(labels_str)
        
        rgb_preview = None
        if 'rgb_preview' in f:
            rgb_preview = f['rgb_preview'][:]
            
    print(f"HDF5 preview and metadata loaded successfully from {filepath}")
    return metadata, labels, rgb_preview

# Insert this function into core/file_io.py

# In core/file_io.py, REPLACE the existing update_h5_roi_settings function.

def update_h5_roi_mask_and_settings(filepath: str, roi_mask: np.ndarray, roi_settings: Dict[str, Any]):
    """
    Efficiently saves or updates the ROI mask (as a dataset) and its
    generation settings (as an attribute) in an existing HDF5 file.

    Args:
        filepath (str): The path to the HDF5 file to update.
        roi_mask (np.ndarray): The boolean ROI mask array to save.
        roi_settings (Dict[str, Any]): The dictionary of ROI settings to save.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found at: {filepath}")

    try:
        with h5py.File(filepath, 'r+') as f:
            # Save the mask as a dataset, deleting the old one if it exists.
            if 'roi_mask' in f:
                del f['roi_mask']
            f.create_dataset('roi_mask', data=roi_mask, compression='gzip')

            # Save the settings as an attribute.
            f.attrs['roi_settings'] = json.dumps(roi_settings)
        print(f"Successfully updated ROI mask and settings in {filepath}")
    except Exception as e:
        print(f"Error updating ROI in {filepath}: {e}")
        raise