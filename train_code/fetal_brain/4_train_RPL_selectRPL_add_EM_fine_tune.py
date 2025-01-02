# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import h5py
from scipy.ndimage import zoom
import torch
import torch.optim as optim
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.net_factory import net_factory
from utils.losses import WeightedDiceLoss , WeightedCrossEntropyLoss , WeightedEMLoss
from dataloaders.fetal_brain_dataset_RPL_selectRPL_UMviaEntropy import BaseDataSet, ValToTensor, TrainToTensor
from utils.val_2D import test_single_volume, test_single_volume_ds
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/data/Liuxy/Code/LXY_RPL_SFDA/data/fetal_brain', help='The root_path of Experiment data file ')
parser.add_argument('--exp', type=str,
                    default='1_RPL_selectRPL_add_EM_target_only', help='model_name') 
parser.add_argument('--net', type=str,
                    default='unet2d', help='net_name') 
parser.add_argument('--data_name', type=str,
                    default='fetal_brain', help='The name of data') 
parser.add_argument('--Domain_args', type=str,
                    default='target', help='Domain_name_idx') 
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network') 
# epoch = max_iterations/n
parser.add_argument('--max_iterations', type=int,
                    default=3000, help='maximum epoch number to train')
parser.add_argument('--max_iterations_name', default="2000", type=str,
                    help='file name of the max_iterations')
parser.add_argument('--lameta_fix', type=float,
                    default=0.1, help='a of weight map')
parser.add_argument('--lameta_fix_name', default="0.1", type=str,
                    help='file name of the a')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-3,
                    help='segmentation network learning rate')
parser.add_argument('--base_lr_name', default="0.001",
                    type=str, help='file for lr')
parser.add_argument('--lr_gamma', type=float,  default=0.5,
                    help='Multiplicative factor of learning rate decay') 
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training') 
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=2022, help='random seed')
parser.add_argument('--threshold', type=float,
                    default=0.6, help='threshold of uncertainty map')
parser.add_argument('--threshold_name', default="0.6", type=str,
                    help='file name of the threshold ')
parser.add_argument('--source_model', type=str,
                    default='', help='source_model_name')  
args = parser.parse_args()

def reliability_based_threshold(uncertainty_map, threshold):
    binary_mask = (uncertainty_map > threshold).float()
    return binary_mask

# ------------------------------  training ------------------------------ #
def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    lameta_fix = args.lameta_fix
    threshold = args.threshold

    model = net_factory(net_type=args.net, in_chns=1, class_num=num_classes)
    model.load_state_dict(torch.load('{0:}'.format(
        args.source_model), map_location='cuda:0'))

    db_train = BaseDataSet(base_dir=args.root_path, 
                           split="train",
                        num=None, 
                        Domain_args = args.Domain_args,
                        transform=transforms.Compose([
                            TrainToTensor(args.patch_size) 
                        ]))
    db_val = BaseDataSet(base_dir=args.root_path,
                         split="val", 
                         Domain_args = args.Domain_args,
                         transform=transforms.Compose([
                             ValToTensor()
                         ]))
                         

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    model.train()

    optimizer = optim.Adam(model.parameters(),
                           lr=base_lr,
                           betas=(0.5, 0.9),
                           weight_decay=1e-5)

    ce_loss = WeightedCrossEntropyLoss()
    dice_loss = WeightedDiceLoss(num_classes)
   
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    # ================================================  training  ================================================
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch, uncertainty_map_batch = sampled_batch[
                'image'], sampled_batch['label'], sampled_batch['uncertainty_map']
            volume_batch, label_batch, uncertainty_map_batch = volume_batch.cuda(
            ), label_batch.cuda(), uncertainty_map_batch.cuda()
 
            final_threshold = threshold * 0.69315     
            UM1_mask = reliability_based_threshold(uncertainty_map_batch, final_threshold)
            total_weights_map_reliable_mask = 1.0 - UM1_mask  
            total_weights_map_unreliable_mask =  UM1_mask 
            total_weights_map_reliable_mask = total_weights_map_reliable_mask.cuda()
            total_weights_map_unreliable_mask = total_weights_map_unreliable_mask.cuda()

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(
                outputs, label_batch[:].long(), total_weights_map_reliable_mask)
            loss_dice = dice_loss(
                outputs_soft, label_batch.unsqueeze(1), weight_map = total_weights_map_reliable_mask)
            sup_loss = 0.5 * (loss_dice + loss_ce)

            un_sup_loss = WeightedEMLoss(outputs_soft, total_weights_map_unreliable_mask)

            loss = sup_loss + lameta_fix * un_sup_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (args.lr_gamma) ** ( iter_num // 1000 )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_train_loss', loss, iter_num)
            writer.add_scalar('info/train_loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/train_loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/train_loss_em', un_sup_loss, iter_num)
            logging.info(
                'iteration %d : total_train_loss : %f, train_loss_ce: %f, train_loss_dice: %f , train_loss_em: %f , lameta_fix : %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), un_sup_loss.item() , lameta_fix ))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            # ================================================  valid  ================================================
            if iter_num > 0 and iter_num % 20 == 0:
                model.eval()
                metric_list = 0.0
                metric_dice_all = []
                metric_hd95_all = []
                metric_assd_all = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["gt"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                    metric_dice_all.append(metric_i[0][0])
                    metric_hd95_all.append(metric_i[0][1])
                    metric_assd_all.append(metric_i[0][2])

                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)
                    writer.add_scalar('info/val_{}_assd'.format(class_i+1),
                                      metric_list[class_i, 2], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                std_performance = np.std(metric_dice_all)

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                std_hd95 = np.std(metric_hd95_all)

                mean_assd = np.mean(metric_list, axis=0)[2]
                std_assd = np.std(metric_assd_all)

                writer.add_scalar('info/val_mean_dice', performance , iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95 , iter_num)
                writer.add_scalar('info/val_mean_assd', mean_assd, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 6))) 
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.net))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'valid data for iteration %d : mean_dice : %f ± %f mean_hd95 : %f ± %f mean_assd : %f ± %f ' % (iter_num, performance, std_performance, mean_hd95, std_hd95, mean_assd, std_assd))
                model.train()

            if iter_num % 200 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../1_RPL_selectRPL_add_EM/{0:}/{1:}/{2:}/{3:}/{4:}/{5:}".format(
        args.exp, args.Domain_args, args.max_iterations_name, args.base_lr_name, args.threshold_name, args.lameta_fix_name)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig( filename = snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    print("training the train(args, snapshot_path)")
    train(args, snapshot_path)