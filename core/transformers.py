# # transformers.py

# """
# Utility functions for applying saved calibration data to images.
# This module contains the core transformation logic, separated from the GUI.
# """

# import cv2
# import numpy as np
# import json
# import os

# def apply_straightening(image_path: str, straighten_data: dict) -> np.ndarray:
#     """
#     Applies the selected correction method (rotation or shear) and optional 
#     translation to an image file.

#     Args:
#         image_path (str): The full path to the image to be corrected.
#         straighten_data (dict): The calibration dictionary from the straightening step.

#     Returns:
#         np.ndarray: The corrected image as a NumPy array.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
#     if img is None:
#         raise FileNotFoundError(f"Could not load image at {image_path}")
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     h, w = img.shape
    
#     method = straighten_data.get("selected_method", "shear") 
#     apply_translation = straighten_data.get("apply_translation", True)
#     print(f"Applying straightening. Method: '{method}', Centering: {apply_translation}")

#     final_img = None
#     if method == "rotation":
#         correction_info = straighten_data.get("rotation_correction", {})
#         if not correction_info:
#             raise ValueError("Rotation correction data not found in calibration dictionary.")
        
#         rot_matrix = np.array(correction_info["rotation_matrix"])
#         rotated_img = cv2.warpAffine(img, rot_matrix, (w, h))
        
#         if apply_translation:
#             trans_matrix = np.array(correction_info["translation_matrix"])
#             final_img = cv2.warpAffine(rotated_img, trans_matrix, (w, h))
#         else:
#             final_img = rotated_img
    
#     elif method == "shear":
#         shear_info = straighten_data.get("shear_correction", {})
#         if not shear_info:
#             raise ValueError("Shear correction data not found in calibration dictionary.")
        
#         avg_angle_deg = shear_info['avg_angle_deg_for_calc']
#         avg_x_pos = shear_info['avg_x_pos_for_calc']
#         avg_angle_rad = np.deg2rad(avg_angle_deg)
#         tan_val = np.tan(avg_angle_rad)
#         slope_dx_dy = -1.0 / tan_val if abs(tan_val) > 1e-9 else 1e9
        
#         sheared_img = np.zeros_like(img)
#         image_center_y = h // 2
        
#         for y in range(h):
#             x_on_tilted_line = avg_x_pos + (y - image_center_y) * slope_dx_dy
#             shift = avg_x_pos - x_on_tilted_line
#             row_data = img[y, :]
#             int_shift = int(round(shift))
#             if int_shift > 0:
#                 if int_shift < w:
#                     sheared_img[y, int_shift:] = row_data[:w - int_shift]
#             elif int_shift < 0:
#                 abs_shift = abs(int_shift)
#                 if abs_shift < w:
#                     sheared_img[y, :w - abs_shift] = row_data[abs_shift:]
#             else:
#                 sheared_img[y, :] = row_data
        
#         if apply_translation:
#             final_translation_needed = (w // 2) - avg_x_pos
#             centering_matrix = np.float32([[1, 0, final_translation_needed], [0, 1, 0]])
#             final_img = cv2.warpAffine(sheared_img, centering_matrix, (w, h))
#         else:
#             final_img = sheared_img
#     else:
#         raise ValueError(f"Unknown straightening method provided in calibration data: '{method}'")
        
#     return final_img

# def apply_cropping(image_to_crop: np.ndarray, crop_data: dict) -> np.ndarray:
#     """
#     Applies cropping to an image array based on calibration data.

#     Args:
#         image_to_crop (np.ndarray): The image (as a NumPy array) to be cropped.
#         crop_data (dict): The calibration dictionary from the cropping step.

#     Returns:
#         np.ndarray: The cropped image as a NumPy array.
#     """
#     bbox = crop_data.get("bbox_pixels")
#     if not bbox:
#         raise ValueError("Bounding box data ('bbox_pixels') not found in calibration dictionary.")
    
#     x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
#     img_h, img_w = image_to_crop.shape[:2]
#     if x + w > img_w or y + h > img_h:
#         w = min(w, img_w - x)
#         h = min(h, img_h - y)

#     return image_to_crop[y:y+h, x:x+w]

# # ### NEW FUNCTION ###
# def map_pixel_to_wavelength(
#     pixel_row: "int | np.ndarray", 
#     spectral_data: dict
# ) -> "float | np.ndarray":
#     """
#     Converts a spectral pixel row index to its corresponding wavelength using
#     the polynomial coefficients from a spectral calibration.

#     This function can handle a single integer or a NumPy array of pixel rows.

#     Args:
#         pixel_row (int or np.ndarray): The pixel row index (or indices) to convert.
#         spectral_data (dict): The calibration dictionary from the spectral step.
#                               Must contain the key "coefficients".

#     Returns:
#         float or np.ndarray: The calculated wavelength (or wavelengths).

#     Raises:
#         ValueError: If the 'coefficients' key is not found in the spectral_data.
#     """
#     coeffs = spectral_data.get("coefficients")
#     if coeffs is None:
#         raise ValueError("Spectral calibration data must contain a 'coefficients' key.")

#     # np.poly1d creates a polynomial function from the coefficients.
#     # This is a highly efficient and convenient way to evaluate the polynomial.
#     poly_func = np.poly1d(coeffs)
    
#     return poly_func(pixel_row)

# # --- EXAMPLE USAGE DEMONSTRATION ---
# if __name__ == '__main__':
#     # This block demonstrates how another script would use the functions in this module.
    
#     print("--- Testing transformers.py functions ---")

#     # 1. Create a dummy master_calibration.json file for the test
#     # In a real application, this file would be loaded.
#     dummy_calibration = {
#         "wizard_version": "1.0",
#         "creation_date": "2023-01-01 12:00:00",
#         "calibration_steps": {
#             "spectral": {
#                 "description": "Wavelength map. Wavelength = f(pixel_row).",
#                 "poly_degree": 2,
#                 # Example: Wavelength = 0.001*p^2 + 0.5*p + 400
#                 "coefficients": [0.001, 0.5, 400.0] 
#             }
#         }
#     }
    
#     # Get the directory of the current script to save the dummy file
#     script_dir = os.path.dirname(__file__)
#     dummy_file_path = os.path.join(script_dir, "dummy_master_calibration.json")
#     with open(dummy_file_path, 'w') as f:
#         json.dump(dummy_calibration, f, indent=4)

#     print(f"Created a dummy calibration file at: {dummy_file_path}")

#     # 2. Load the master calibration file and extract the spectral data
#     with open(dummy_file_path, 'r') as f:
#         master_cal = json.load(f)
    
#     spectral_cal_data = master_cal["calibration_steps"]["spectral"]
    
#     # 3. Use the new map_pixel_to_wavelength function
    
#     # Test with a single pixel row
#     pixel = 100
#     wavelength = map_pixel_to_wavelength(pixel, spectral_cal_data)
#     print(f"\nFor a single pixel:")
#     print(f"  Pixel Row {pixel} -> Wavelength {wavelength:.2f} nm")

#     # Test with a NumPy array of pixel rows (e.g., to create a wavelength axis)
#     pixel_axis = np.arange(0, 5) # Rows 0, 1, 2, 3, 4
#     wavelength_axis = map_pixel_to_wavelength(pixel_axis, spectral_cal_data)
#     print(f"\nFor an array of pixels:")
#     for p, w in zip(pixel_axis, wavelength_axis):
#         print(f"  Pixel Row {p} -> Wavelength {w:.2f} nm")
        
#     # Clean up the dummy file
#     os.remove(dummy_file_path)
#     print(f"\nCleaned up dummy file.")

# transformers.py

# transformers.py (Modified)

"""
Utility functions for applying saved calibration data to images.
This module contains the core transformation logic, separated from the GUI.
"""

import cv2
import numpy as np
import json
import os
from typing import Union

# --- Private Helper Functions for Straightening ---
# ... (These helpers _apply_rotation_transform and _apply_shear_transform are unchanged) ...
def _apply_rotation_transform(
    img: np.ndarray, 
    correction_info: dict, 
    apply_translation: bool
) -> np.ndarray:
    """Applies rotation and optional translation to an image."""
    h, w = img.shape[:2]
    rot_matrix = np.array(correction_info.get("rotation_matrix"))
    if rot_matrix is None:
        raise ValueError("Rotation matrix not found in calibration data.")
        
    transformed_img = cv2.warpAffine(img, rot_matrix, (w, h))
    
    if apply_translation:
        trans_matrix = np.array(correction_info.get("translation_matrix"))
        if trans_matrix is None:
            raise ValueError("Translation matrix not found for rotation.")
        transformed_img = cv2.warpAffine(transformed_img, trans_matrix, (w, h))
        
    return transformed_img

def _apply_shear_transform(
    img: np.ndarray, 
    shear_info: dict, 
    apply_translation: bool
) -> np.ndarray:
    """
    Applies a shear transformation to correct for line tilt, followed by
    an optional centering translation. This is done in a single, efficient
    affine transformation.
    """
    h, w = img.shape[:2]
    
    avg_angle_deg = shear_info.get('avg_angle_deg_for_calc')
    avg_x_pos = shear_info.get('avg_x_pos_for_calc')
    if avg_angle_deg is None or avg_x_pos is None:
        raise ValueError("Shear calculation parameters not found in calibration data.")

    avg_angle_rad = np.deg2rad(avg_angle_deg)
    tan_val = np.tan(avg_angle_rad)
    slope_dx_dy = -1.0 / tan_val if abs(tan_val) > 1e-9 else 1e9
    
    shear_factor = -slope_dx_dy
    image_center_y = h // 2
    shear_offset = image_center_y * slope_dx_dy
    translation_offset = (w // 2) - avg_x_pos if apply_translation else 0
    total_offset_x = shear_offset + translation_offset
    
    transform_matrix = np.float32([
        [1, shear_factor, total_offset_x],
        [0, 1,            0]
    ])
    
    return cv2.warpAffine(img, transform_matrix, (w, h))


# --- Public API Functions ---

def apply_straightening_to_image_array(
    img: np.ndarray, 
    straighten_data: dict
) -> np.ndarray:
    """
    Applies straightening to an image array. This is the preferred function for
    live view processing as it avoids disk I/O.

    Args:
        img (np.ndarray): The image array to be corrected.
        straighten_data (dict): The calibration dictionary from the straightening step.

    Returns:
        np.ndarray: The corrected image as a NumPy array.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    method = straighten_data.get("selected_method", "shear") 
    apply_translation = straighten_data.get("apply_translation", True)

    if method == "rotation":
        correction_info = straighten_data.get("rotation_correction")
        if not correction_info:
            raise ValueError("Rotation correction data not found in calibration dictionary.")
        return _apply_rotation_transform(img, correction_info, apply_translation)
    
    elif method == "shear":
        shear_info = straighten_data.get("shear_correction")
        if not shear_info:
            raise ValueError("Shear correction data not found in calibration dictionary.")
        return _apply_shear_transform(img, shear_info, apply_translation)
        
    else:
        raise ValueError(f"Unknown straightening method: '{method}'")

def apply_straightening(image_path: str, straighten_data: dict) -> np.ndarray:
    """
    Applies straightening to an image file.
    (This function is preserved for compatibility but is less efficient for live view).
    """
    img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"Could not load or decode image at {image_path}")
    return apply_straightening_to_image_array(img, straighten_data)

# ... (The rest of the file: apply_cropping, map_pixel_to_wavelength, and __main__ are unchanged) ...
def apply_cropping(image_to_crop: np.ndarray, crop_data: dict) -> np.ndarray:
    """
    Applies cropping to an image array based on calibration data.

    Args:
        image_to_crop (np.ndarray): The image (as a NumPy array) to be cropped.
        crop_data (dict): The calibration dictionary from the cropping step.

    Returns:
        np.ndarray: The cropped image as a NumPy array.
    """
    bbox = crop_data.get("bbox_pixels")
    if not bbox:
        raise ValueError("Bounding box data ('bbox_pixels') not found in calibration dictionary.")
    
    x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
    
    img_h, img_w = image_to_crop.shape[:2]
    
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    return image_to_crop[y1:y2, x1:x2]

def map_pixel_to_wavelength(
    pixel_row: Union[int, np.ndarray], 
    spectral_data: dict
) -> Union[float, np.ndarray]:
    """
    Converts a spectral pixel row index to its corresponding wavelength using
    the polynomial coefficients from a spectral calibration.
    """
    coeffs = spectral_data.get("coefficients")
    if coeffs is None:
        raise ValueError("Spectral calibration data must contain a 'coefficients' key.")

    poly_func = np.poly1d(coeffs)
    
    return poly_func(pixel_row)