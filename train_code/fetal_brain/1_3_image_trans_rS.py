import os
import nibabel as nib
import numpy as np
import torch
from scipy.optimize import minimize
import csv

def load_nii_file(file_path):
    # Load the NIfTI file
    img = nib.load(file_path)
    # Get the image data as a numpy array
    img_data = img.get_fdata()
    return img, img_data

def process_case(img_data):
    # Remove background where pixel value is -1
    mask = img_data != 0
    img_data = img_data[mask]
    # Normalize to [0, 1] range
    img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
    return img_data, mask

def gamma_correction(image, gamma):
    # Adding small constant to avoid divide by zero error
    return np.power(image + 1e-10, gamma)

def objective(gamma, image, target_mean, target_std):
    corrected_image = gamma_correction(image, gamma)
    corrected_mean = np.mean(corrected_image)
    corrected_std = np.std(corrected_image)
    return abs((corrected_mean - target_mean) + (corrected_std - target_std))
    # return (corrected_mean - target_mean)**2 + (corrected_std - target_std)**2

def main():
    root_dir = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01'
    data_dir = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source'
    output_dir = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source_transr_mean0.5_std0.29'
    output_csv = os.path.join(root_dir, 'source_transr_mean0.5_std0.29_case_statistics.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_image.nii.gz')]

    with open(output_csv, mode='w', newline='') as csv_file:
        fieldnames = ['case_name', 'initial_mean', 'initial_std', 'gamma', 'corrected_mean', 'corrected_std']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for file_path in files:
            case_name = os.path.basename(file_path)
            img, img_data = load_nii_file(file_path)
            original_shape = img_data.shape
            processed_data, mask = process_case(img_data)
            initial_mean = np.mean(processed_data)
            initial_std = np.std(processed_data)

            target_mean = 0.5
            target_std = 0.29

            initial_gamma = 1.0
            result = minimize(objective, initial_gamma, args=(processed_data, target_mean, target_std), method='Nelder-Mead')
            optimal_gamma = result.x[0]

            corrected_data = gamma_correction(processed_data, optimal_gamma)
            corrected_mean = np.mean(corrected_data)
            corrected_std = np.std(corrected_data)

            # Create the final 3D image with corrected values
            final_data = np.zeros(original_shape)
            final_data[mask] = corrected_data

            # Save the corrected data as a new NIfTI file
            new_img = nib.Nifti1Image(final_data, img.affine, img.header)
            new_file_path = os.path.join(output_dir, case_name)
            nib.save(new_img, new_file_path)

            writer.writerow({
                'case_name': case_name,
                'initial_mean': initial_mean,
                'initial_std': initial_std,
                'gamma': optimal_gamma,
                'corrected_mean': corrected_mean,
                'corrected_std': corrected_std
            })

            print(f'Processed {case_name}: initial_mean = {initial_mean}, initial_std = {initial_std}, gamma = {optimal_gamma}, corrected_mean = {corrected_mean}, corrected_std = {corrected_std}')

if __name__ == "__main__":
    main()
