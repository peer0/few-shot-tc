import torch
import torch.nn as nn


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model #원본 모델
        self.decay = decay #decay 계수
        self.shadow = {} # 파라미터의 EMA 값을 저장할 딕셔너리
        self.backup = {} # 원본 모델의 파라미터를 임시로 백업할 딕셔너리

    def load(self, ema_model):
        # EMA 값을 불러와 설정
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()# EMA 값을 현재 파라미터로 설정

    def register(self):
        # 원본 모델의 파라미터를 EMA에 등록
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()# 현재 파라미터를 EMA 값으로 등록

    def update(self):
         # EMA 값을 업데이트
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                # 새로운 EMA 값 계산
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name] #모름
                self.shadow[name] = new_average.clone() # 새로운 EMA 값으로 업데이트

    def apply_shadow(self):
        # EMA 값을 모델에 적용
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                 # 모델 파라미터를 EMA 값으로 설정하고, 백업에 현재 파라미터 저장
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        # 원본 모델로 복원
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                # 모델 파라미터를 원본으로 복원
                param.data = self.backup[name]
        self.backup = {}# 백업 초기화