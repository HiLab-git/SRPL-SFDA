export PYTHONPATH=$PYTHONPATH: train_code
# (a) Target-domain pseudo-label generation with Test-Time Tri-branch Intensity Enhancement
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/1_1_image_trans_equal.py ;
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/1_2_image_trans_rD.py ;
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/1_3_image_trans_rS.py ;
#  input source-model to seg...
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/1_4_average_image_transr_equal_pl.py ;

# (b) Refined Pseudo-Labels by SAM
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/2_1_everything_bbox.py ;
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/2_2_concat_image.py ;
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/2_3_Med_SAM_bbox_seg.py ;

# (c) Reliable Pseudo Label Selection
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/3_train_RPL_selectRPL_fine_tune.py ;

# (d) Reliability-Aware Pseudo-Label Supervision and Regularization
CUDA_VISIBLE_DEVICES=0,1 python train_code/fetal_brain/4_train_RPL_selectRPL_add_EM_fine_tune.py ;
