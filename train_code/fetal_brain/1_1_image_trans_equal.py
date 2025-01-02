import os
import numpy as np
import nibabel as nib
import cv2

def load_nii(file_path):
    return nib.load(file_path)

def save_nii(image, file_path):
    nib.save(image, file_path)

def apply_histogram_equalization(image_data):
    # Apply histogram equalization to each slice
    equalized_slices = []
    for slice_idx in range(image_data.shape[2]):
        slice_np = image_data[:, :, slice_idx]
        # Convert to uint8 and normalize to [0, 255]
        slice_np = (slice_np * 255).astype(np.uint8)

        # Apply histogram equalization using OpenCV
        equalized_slice = cv2.equalizeHist(slice_np)
        # Normalize back to [0, 1] if necessary
        equalized_slice = equalized_slice.astype(np.float32) / 255.0

        equalized_slices.append(equalized_slice)
    
    # Stack the slices back into a 3D volume
    equalized_volume = np.stack(equalized_slices, axis=2)
    
    return equalized_volume

source_dir = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source'
dest_dir = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01_equal/source'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for file_name in os.listdir(source_dir):
    if file_name.endswith('.nii.gz'):
        file_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        
        nii_image = load_nii(file_path)
        image_data = nii_image.get_fdata()

        if file_name.endswith('_image.nii.gz'):
            equalized_data = apply_histogram_equalization(image_data)
            new_nii_image = nib.Nifti1Image(equalized_data, nii_image.affine, nii_image.header)
        elif file_name.endswith('_label.nii.gz'):
            new_nii_image = nii_image
        else:
            continue
        
        save_nii(new_nii_image, dest_path)
        print(f"Processed and saved: {file_name}")
