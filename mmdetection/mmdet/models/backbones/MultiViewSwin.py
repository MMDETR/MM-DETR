import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones.swin import SwinTransformer  # Import Swin Transformer

@MODELS.register_module()
class MultiViewSwin(SwinTransformer):
    def __init__(self, embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], **kwargs):
        super().__init__(embed_dims=embed_dims, depths=depths, num_heads=num_heads, **kwargs)
    
    def forward(self, inputs):
        mlo_imgs = inputs[:, 0, :, :, :]
        cc_imgs = inputs[:, 1, :, :, :]
        mlo_features = super().forward(mlo_imgs)
        cc_outs = super().forward(cc_imgs)

        return mlo_features, cc_outs
