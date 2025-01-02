#  the fusion images of 3 channels
import os
import nibabel as nib
import h5py
import numpy as np
import SimpleITK as sitk
from PIL import Image
def load_nii(file_path):
    img_obj = sitk.ReadImage(file_path)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    return sitk.GetArrayFromImage(img_obj)

def fusion_image_png(base_path, path1, path2, output_path):
    for filename in os.listdir(base_path):
        if filename.endswith("_image.nii.gz"):
            base_file = os.path.join(base_path, filename)
            file1 = os.path.join(path1, filename)
            file2 = os.path.join(path2, filename)

            # load NIfTI file
            base_img = load_nii(base_file)
            img1 = load_nii(file1)
            img2 = load_nii(file2)

            # normalize image
            image_min_value = np.min(base_img)
            image_max_value = np.max(base_img)
            base_img = (base_img - image_min_value) / (image_max_value - image_min_value)

            for i in range(base_img.shape[0]): 
                slice_base = base_img[i, :, :]
                slice1 = img1[i, :, :]
                slice2 = img2[i, :, :]
                slice_base = np.uint8(slice_base * 255)
                slice1 = np.uint8(slice1 * 255)
                slice2 = np.uint8(slice2 * 255)

                inputs =  np.asarray([slice_base, slice1, slice2])   
                inputs = np.transpose(inputs, (1, 2, 0))
                inputs_image = Image.fromarray(inputs.astype('uint8'), 'RGB')

                output_filename = os.path.join(output_path, f"{filename[:-7]}_slice{i}.png")
                inputs_image.save(output_filename)
    print("finish")


if __name__ == '__main__':
    fetal_brain_image_norm_path = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source'
    fetal_brain_Dr_path = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source_transr_mean0.5_std0.29'
    fetal_brain_Gr_path = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source_transr_0.5'
    fetal_brain_equal_path = '/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01_equal/source'
    fetal_brain_transrTwo_equal_path = '../image/0_image_concat/6_3_transrTwo_equal'

    os.makedirs(fetal_brain_transrTwo_equal_path, exist_ok=True)
    fusion_image_png(fetal_brain_Dr_path,  fetal_brain_Gr_path , fetal_brain_equal_path, fetal_brain_transrTwo_equal_path) 

    print("finish all")

