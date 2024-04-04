# Criterions
import torch 
import torch.nn as nn 
from torch.nn import functional as F


def ce_loss(logits, targets, reduction='mean'):
    # logits : 모델이 예측한 클래스의 로그확률
    # targets : 정답 클래스
    # reduction은 손실 값들을 어떻게 집계할지: 'none', 'mean'(기본값), 'sum' 중 하나를 선택할 수 있습니다.
    # cross entropy loss in pytorch.
    # logit과 target의 형태가 같다면, 
    # 
    if logits.shape == targets.shape:
        #1.target은 원핫인코딩된 레이블로 간주하고, 
        log_pred = F.log_softmax(logits, dim=-1)
        #2.각 클래스에 대한 음의 로그 가능도(negative lof liklihood NLL) 계산
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else: #4.두 텐서의 형태가 다르다면, target은 클래스 인덱스를 포함하고 있다고 간주하고,
        log_pred = F.log_softmax(logits, dim=-1)
        #3.그리고 이 값을 reduction 인자에 따라 집계합니다.
        #5.PyTorch의 nll_loss 함수를 사용하여 NLL 손실을 계산합니다.
        #5.(이 함수는 logits를 소프트맥스 함수를 통과시켜 확률로 변환하고, 이 확률에 대한 targets의 NLL을 계산합니다.)
        return F.nll_loss(log_pred, targets, reduction=reduction)



def consistency_loss(logits, target, loss_type='ce', mask=None, disagree_weight_masked=None):
    # consistency loss
    # input: logits_x_ulb_s, pseudo_label, loss_type, mask
    if loss_type == 'mse':
        loss = F.mse_loss(logits, target, reduction='none')
    else:
        loss = ce_loss(logits, target, reduction='none')

    if mask is not None:
        # add pseudo-label mask
        loss = loss * mask
        if disagree_weight_masked is not None:
            # add disagreement weight and mask
            loss = loss * disagree_weight_masked

    return loss.mean()
