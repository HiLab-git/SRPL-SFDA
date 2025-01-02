import os
import numpy as np
import SimpleITK as sitk
import torch
from skimage.io import imread,imsave
import gc
import matplotlib.pyplot as plt 
import os
import numpy as np
import torch
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import h5py
from PIL import Image
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
curPath = os.path.abspath(os.path.dirname(
    "/home/data/Liuxy/Code/LXY_RPL_SFDA/codesfda/"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.util_mideepseg import get_Centroid_Endpoint_seed_points_2d

# # - - - - - - - - - - - - - - - bbox mode - - - - - - - - - - - - - - - - - - 
def show_mask(save_path, mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    save_path = "{0:}_score0.png".format(save_path.replace("_addpoint.png", "") )
    ax.imshow(mask_image)
    plt.savefig(save_path)
    del ax
    gc.collect()

def show_box(save_path, box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    
    plt.savefig(save_path)
    del ax
    gc.collect()

def SAM_bbox_mode(base_h5_path, image_concatpng_path, output_3Dnii_path , output_png_path , image_nii_base_path):
    for filename in os.listdir(base_h5_path):
        if filename.endswith('.h5'):
            print(filename)
            with h5py.File(os.path.join(base_h5_path, filename), 'r') as h5f:
                image = h5f['image'][:]
                image = (image + 1) / 2.0
                image = (image * 255).astype(np.float32)
                label = h5f['label'][:]
                total_pseudo_label = np.zeros_like(image)
            for slice_index in range(image.shape[0]):
                slice_image = image[slice_index, :, :]
                slice_label = label[slice_index, :, :]
                if slice_label.sum() == 0 :
                    total_pseudo_label[slice_index] = slice_label
                else:
                    image_concatpng_path_slice = "{0:}/{1:}{2:}{3:}".format(image_concatpng_path, filename.replace(".h5", "_slice"), slice_index, ".png")  
                    image_png = imread(image_concatpng_path_slice)[:,:,:3]
                    predictor.set_image(image_png)
                    Centroid_points, Endpoint_points = get_Centroid_Endpoint_seed_points_2d(slice_label) 
                    (lefty , leftx) , (righty , rightx)= Endpoint_points[0] , Endpoint_points[3]
                    image_addpoint_path_slice = "{0:}/{1:}{2:}{3:}".format(output_png_path, filename.replace(".h5", "_slice"), slice_index, "_addbbox.png")
                    input_box = np.array([leftx , lefty , rightx , righty ])
                    masks, _, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_png)
                    plt.axis('off')
                    show_box(image_addpoint_path_slice,input_box, plt.gca())
                    show_mask( image_addpoint_path_slice, masks[0], plt.gca())
                    plt.show()
                    mask_float32 = masks[0].astype(np.float32)
                    total_pseudo_label[slice_index] = mask_float32
            plt.close('all')
            image_nii_base_path = "{0:}".format(image_nii_base_path)
            image_nii_path = "{0:}/{1:}".format(image_nii_base_path, filename.replace(".h5", ".nii.gz"))
            img_obj = sitk.ReadImage(image_nii_path)
            spacing = img_obj.GetSpacing()
            origin = img_obj.GetOrigin()
            direction = img_obj.GetDirection()

            plmask_itk = sitk.GetImageFromArray(total_pseudo_label.astype(np.float32))
            plmask_itk.SetSpacing(spacing)
            plmask_itk.SetOrigin(origin)
            plmask_itk.SetDirection(direction)
            sitk.WriteImage(plmask_itk, output_3Dnii_path + "/" + filename.replace(".h5", "_pplabel.nii.gz") )
        print("finish")

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    print("finish_all")


if __name__ == '__main__':
    sam_checkpoint = "/home/data/Liuxy/Code/LXY_RPL_SFDA/codesfda/segment_anything/checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    mask_generator_everything = SamAutomaticMaskGenerator(model=sam, points_per_side=12, pred_iou_thresh=0.7, stability_score_thresh=0.7, crop_n_layers=1, crop_n_points_downscale_factor=2, min_mask_region_area=100, )
    predictor = SamPredictor(sam)

    # # fetal_brain
    fetal_brain_image_nii_base_path = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_01/source"
    fetal_brain_h5_path = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_K_NL_pre_avePL/fetal_brain_norm_volumes/setting1_NL_trans/source"
    
    fetal_brain_image_init_path3 = '../image/0_image_concat/6_3_transrTwo_equal'
    fetal_brain_SAMbbox_output_image_init_path3 = '../image/1_image_concat_SAM_seg_results/2_add_bbox_promt/6_3_transrTwo_equal'
    fetal_brain_out_png_image_init3 = "../image/1_image_concat_SAM_seg_results/2_add_bbox_promt/6_3_transrTwo_equal/png"
    os.makedirs(fetal_brain_SAMbbox_output_image_init_path3, exist_ok=True)
    os.makedirs(fetal_brain_out_png_image_init3, exist_ok=True)
    SAM_bbox_mode(fetal_brain_h5_path, fetal_brain_image_init_path3, fetal_brain_SAMbbox_output_image_init_path3 , fetal_brain_out_png_image_init3 , fetal_brain_image_nii_base_path ) 

    print("finish all")
  
