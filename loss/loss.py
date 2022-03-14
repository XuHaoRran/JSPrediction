import torch
import torch.nn as nn
from .classification import ClassificationLoss
from .dice import EDiceLoss, EDiceLoss_Val
from .survival import SurvivalLoss

class JointLoss(nn.Module):
    """JointLoss = SegLoss + SurvLoss"""
    def __init__(self, mode='train', device='gpu'):
        super(JointLoss, self).__init__()



        if mode == 'train':
            self.SegLoss = EDiceLoss()
        else:
            self.SegLoss = EDiceLoss_Val()

        self.SurvLoss = SurvivalLoss()
        self.EDiceLoss_Val = EDiceLoss_Val()

        if device == 'gpu':
            self.SegLoss, self.SurvLoss, self.EDiceLoss_Val = self.SegLoss.cuda(), self.SurvLoss.cuda(), self.EDiceLoss_Val.cuda()

    def forward(self, pred_dict, label_dict):

        seg_p = pred_dict['seg']
        surv_p = pred_dict['surv']

        seg_l = label_dict['seg']
        surv_l = label_dict['surv']

        seg_loss = self.SegLoss(seg_p, seg_l)
        surv_loss = self.SurvLoss(surv_p, surv_l)

        joint_loss = seg_loss + surv_loss

        # joint_loss = self.SegLoss(seg_p, seg_l) + self.SurvLoss(surv_p, surv_l) + self.ClsLoss(cls_p, cls_l)

        return joint_loss, seg_loss, surv_loss

