# -*- coding: utf-8 -*-
from __future__ import print_function, division

from networks.unet import UNet
import os,sys
root_path = os.getcwd()
sys.path.insert(0,root_path)

def net_factory(net_type="unet2d", in_chns=1, class_num=2 ):
    if net_type == "unet2d":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()

    else:
        net = None
    return net
