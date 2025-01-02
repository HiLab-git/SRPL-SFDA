import os
import glob
import SimpleITK as sitk
import torch
import numpy as np

# Paths
path1 = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_with_SAM/1_Corner_point_detection/setting1_NL_trans/source/image/1_image_concat_SAM_seg_results/2_add_bbox_promt/image_rD"
path2 = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_with_SAM/1_Corner_point_detection/setting1_NL_trans/source/image/1_image_concat_SAM_seg_results/2_add_bbox_promt/image_rS"
path3 = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_with_SAM/1_Corner_point_detection/setting1_NL_trans/source/image/1_image_concat_SAM_seg_results/2_add_bbox_promt/image_equal"
output_path = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_with_SAM/1_Corner_point_detection/setting1_NL_trans/source/image/1_image_concat_SAM_seg_results/2_add_bbox_promt/6_all_average"

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Find all files ending with "_image_pplabel.nii.gz" in path1
files = glob.glob(os.path.join(path1, "*_image_pplabel.nii.gz"))

def load_nii(file_path):
    img_obj = sitk.ReadImage(file_path)
    spacing = img_obj.GetSpacing()
    origin = img_obj.GetOrigin()
    direction = img_obj.GetDirection()
    return sitk.GetArrayFromImage(img_obj) , spacing, origin, direction

# Process each file
for file in files:
    # Load images from all three paths
    filename = os.path.basename(file)
    print(filename)
    file2 = os.path.join(path2, filename)
    file3 = os.path.join(path3, filename)

    # Read images using SimpleITK
    image1 ,spacing, origin, direction= load_nii(file)
    image2 , _ , _ , _ = load_nii(file2)
    image3 , _ , _ , _ = load_nii(file3)
    average_prediction = np.zeros_like(image1)
    for ind in range(image1.shape[0]):
        slice1 = image1[ind, :, :]
        slice2 = image2[ind, :, :]
        slice3 = image3[ind, :, :]
        averaged_slice = np.mean([slice1, slice2, slice3], axis=0) 
        threshold = 0.5
        segmented_slice = (averaged_slice > threshold).astype(np.int)
        average_prediction[ind] = segmented_slice
    img_itk = sitk.GetImageFromArray(average_prediction.astype(np.float32))
    img_itk.SetSpacing(spacing)
    img_itk.SetOrigin(origin)
    img_itk.SetDirection(direction)
    sitk.WriteImage(img_itk, output_path + '/' + filename )
print("Processing completed.")
