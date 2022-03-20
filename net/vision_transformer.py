# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
import copy
import logging

import torch
import torch.nn as nn

from .jsp_net import SwinTransformerSys3D

logger = logging.getLogger(__name__)


class SWT_JSP(nn.Module):
    def __init__(self, config, num_classes=1, zero_head=False, embed_dim=96, win_size=7):
        super(SWT_JSP, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.embed_dim = embed_dim
        self.win_size = win_size
        self.win_size = (self.win_size,self.win_size,self.win_size)

        self.swtjsp_net = SwinTransformerSys3D(img_size=(128, 128, 128),
                                            patch_size=(4, 4, 4),
                                            in_chans=3,
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=[2, 2, 2, 1],
                                            depths_decoder=[1, 2, 2, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=self.win_size,
                                            mlp_ratio=4.,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=0.,
                                            attn_drop_rate=0.,
                                            drop_path_rate=0.1,
                                            norm_layer=nn.LayerNorm,
                                            patch_norm=True,
                                            use_checkpoint=False,
                                            frozen_stages=-1)

    def forward(self, x, sub_x):
        logits = self.swtjsp_net(x, sub_x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model_dict = self.swtjsp_net.state_dict()
            pretrained_dict = pretrained_dict['state_dict']
            temp_pretrained_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if 'swin_unet.' in k:
                    k = k.split('swin_unet.')[1]
                temp_pretrained_dict[k] = v
            del pretrained_dict
            temp_model_dict = OrderedDict()
            for k,v in temp_pretrained_dict.items():
                for m in model_dict.keys():
                    if k in m and v.shape == model_dict[m].shape:
                        temp_model_dict[m] = v
            model_dict.update(temp_model_dict)
            self.swtjsp_net.load_state_dict(model_dict, strict=False)
        else:
            print("none pretrain")


