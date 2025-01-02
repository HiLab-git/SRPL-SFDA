# -*- coding: utf-8 -*-
from __future__ import print_function, division
import sys
import argparse
import logging
import os
import shutil
from glob import glob
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain', help='The root_path of Experiment data file ')
parser.add_argument('--SAMseg_path', type=str,
                    default='...', help='The root_path of Experiment data file ')
parser.add_argument('--exp', type=str,
                    default='fetal_brain_norm_01_with_SAM/target_test_data', help='model_name')  
parser.add_argument('--promtwhich', type=str,
                    default='0_pase1_pseudolabel', help='add point or bbox')  
parser.add_argument('--inputwhich', type=str,
                    default='pase1_high_avrage3', help='input three different input') 
parser.add_argument('--data_name', type=str,
                    default='fetal_brain', help='The name of data')  
parser.add_argument('--Domain_args', type=str,
                    default='setting1_NL_trans', help='Domain_name_idx')  
parser.add_argument('--Domain_name', type=str,
                    default='source', help='Domain_name_idx')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')  

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, hd95, asd
    else:
        return 0, 90, 40

def test_single_volume(case, FLAGS):
    h5pre = h5py.File(
        FLAGS.root_path + "/fetal_brain_norm_K_NL_pre_avePL/fetal_brain_norm_volumes/{0:}/source/{1:}.h5".format(FLAGS.Domain_args, case), 'r')
    label = h5pre['gt']
    SAM_label_case = "/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain/fetal_brain_norm_with_SAM/2_K_NL_pre_avePL/setting1_TrEq_sm_Dr_Gr_Equal_high/source/{0:}_label.nii.gz".format(case)
    labelpre_nii = nib.load(SAM_label_case)
    labelpre = labelpre_nii.get_fdata()
    labelpre = np.array(labelpre, dtype=np.int32)  
    if labelpre.shape != label.shape:
        labelpre = labelpre.transpose(2, 1, 0) 

    metric_list = []
    for i in range(1, FLAGS.num_classes):
        metric_list.append(calculate_metric_percase(
            labelpre == i, label == i))

    return metric_list


def Inference(FLAGS):
    with open(FLAGS.root_path + '/data_split/' + FLAGS.Domain_name + '/image_test.csv', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])


    test_save_path = "../source/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    metric_list = 0.0
    metric_dice_all = []
    metric_hd95_all = []
    metric_assd_all = []

    for case in tqdm(image_list):
        metric_i = test_single_volume(case, FLAGS)
        logging.info(
            '{0:} -- data metrics -- metric_dice : {1:} , metric_hd95 : {2:} ,  metric_assd : {3:}'.format(case, metric_i[0][0] , metric_i[0][1] , metric_i[0][2]))

        metric_list += np.array(metric_i)
        metric_dice_all.append(metric_i[0][0])
        metric_hd95_all.append(metric_i[0][1])
        metric_assd_all.append(metric_i[0][2])
    metric_list = metric_list / len(image_list)

    performance = np.mean(metric_list, axis=0)[0]
    std_performance = np.std(metric_dice_all)

    mean_hd95 = np.mean(metric_list, axis=0)[1]
    std_hd95 = np.std(metric_hd95_all)

    mean_assd = np.mean(metric_list, axis=0)[2]
    std_assd = np.std(metric_assd_all)

    logging.info(
        'This is %s test data %s metrics: mean_dice : %f ± %f mean_hd95 : %f ± %f mean_assd : %f ± %f ' % (FLAGS.Domain_name, FLAGS.Domain_args, performance, std_performance, mean_hd95, std_hd95, mean_assd, std_assd))
    return metric_list


def Logging_save_path(FLAGS):
    test_save_path = "../source/"
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    return test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    logging.basicConfig(filename=Logging_save_path(FLAGS) + "label_testlog.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(FLAGS))
    metric = Inference(FLAGS)
    print(metric)