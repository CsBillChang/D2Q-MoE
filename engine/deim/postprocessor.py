"""     
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)     
Copyright(c) 2023 lyuwenyu. All Rights Reserved.  
"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision

from ..core import register

def mod(a, b):
    out = a - a // b * b    
    return out

def scale_boxes_to_original_image(
    boxes: torch.Tensor,   
    orig_target_sizes: torch.Tensor,
    resize_pad: dict | None = None,   
) -> torch.Tensor:
    bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

    if resize_pad is None:
        return bbox_pred * orig_target_sizes.repeat(1, 2).unsqueeze(1)    

    input_sizes = resize_pad['size'].to(device=boxes.device, dtype=boxes.dtype)  
    padding = resize_pad['padding'].to(device=boxes.device, dtype=boxes.dtype)

    padded_boxes = bbox_pred * input_sizes.repeat(1, 2).unsqueeze(1) 
    left_top = padding[:, [0, 1]].unsqueeze(1)
    resized_sizes = input_sizes - padding[:, [0, 1]] - padding[:, [2, 3]]    
    scales = resized_sizes / orig_target_sizes.to(device=boxes.device, dtype=boxes.dtype)    
   
    original_boxes = (padded_boxes - left_top.repeat(1, 1, 2)) / scales.repeat(1, 2).unsqueeze(1) 
    max_xyxy = orig_target_sizes.to(device=boxes.device, dtype=boxes.dtype).repeat(1, 2).unsqueeze(1)  
    return original_boxes.clamp(min=0).minimum(max_xyxy)

@register()     
class PostProcessor(nn.Module):
    # 共享的参数列表
    __share__ = [
        'num_classes',  
        'use_focal_loss',   
        'num_top_queries',     
        'remap_mscoco_category'
    ]

    def __init__(
        self,    
        num_classes=80,  # 目标类别数，默认 80（如 COCO 数据集）    
        use_focal_loss=True,  # 是否使用 Focal Loss     
        num_top_queries=300,  # 选择前 num_top_queries 个目标    
        remap_mscoco_category=False  # 是否对 COCO 数据集的类别进行重新映射
    ) -> None:
        super().__init__()  
        self.use_focal_loss = use_focal_loss  
        self.num_top_queries = num_top_queries     
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False  # 部署模式标志位，默认关闭

    def extra_repr(self) -> str:  
        """ 
        额外的表示方法，在打印模型结构时显示   
        """     
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'  
 
    def forward(self, outputs, orig_target_sizes: torch.Tensor, for_eval=False, resize_pad=None):   
        """
        前向传播函数，处理检测头的输出，将其转换为最终检测结果。  
        :param outputs: 预测的输出字典，包含 'pred_logits'（类别分数）和 'pred_boxes'（边界框）  
        :param orig_target_sizes: 原始图像的尺寸，用于恢复边界框到原始图像尺度 
        :param for_eval: 是否用于验证结果     
        :return: 处理后的检测结果，包含 labels、boxes、scores     
        """  
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']     
   
        bbox_pred = scale_boxes_to_original_image(boxes, orig_target_sizes, resize_pad)  
    
        if self.use_focal_loss:   
            # 使用 Focal Loss 计算类别分数   
            scores = F.sigmoid(logits)
            # 选择最高的 num_top_queries 个目标   
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            # 计算类别标签
            labels = index % self.num_classes  # 取模计算类别索引   
            index = index // self.num_classes  # 计算 box 索引
            # 根据索引选择对应的边界框
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))    
        else:
            # 计算 softmax 归一化后的类别分数
            scores = F.softmax(logits, dim=-1)[:, :, :-1]  # 去掉背景类别 
            scores, labels = scores.max(dim=-1)  # 选择最大分数对应的类别    
            if scores.shape[1] > self.num_top_queries:
                # 选择最高的 num_top_queries 个目标
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))
    
        # 部署模式下，直接返回 labels、boxes、scores 
        if self.deploy_mode:
            return labels, boxes, scores  
    
        # 如果需要映射 COCO 类别索引
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category   
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)    
    
        # 组织输出结果    
        results = []   
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)     
 
        return results
     
    def deploy(self):
        """
        设置模型为部署模式（推理模式）
        """
        self.eval()
        self.deploy_mode = True
        return self
