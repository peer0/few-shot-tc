# A class for a group of networks:
# Functions: 1. initialize the group of networks
#            2. forward the group of networks
#            3. update the group of networks

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel
from utils.ema import EMA
from models.model import TextClassifier



class NetGroup(nn.Module):
    def __init__(self, net_arch, num_nets, n_classes, device, lr, lr_linear=1e-3):
        super(NetGroup, self).__init__()
        # parameters
        self.net_arch = net_arch #신경망 아키텍처
        self.num_nets = num_nets #신경망 개수
        self.n_classes = n_classes #클래스 개수
        self.device = device #디바이스
        self.lr = lr #lr

        # initialize the group of networks
        # 신경망 그룹 초기화
        self.nets = {} #신경망 그룹 저장 딕셔너리
        ##{0: BERTForSequenceClass...}
        ##{1: BERTForSequenceClass...}
        for i in range(num_nets): 
            self.nets[i] = self.init_net(self.net_arch[i]) #초기화

        # initialize optimizers for the group of networks
        #신경망 그룹을 위한 옵티마이저 초기화
        self.optimizers = {} #옵티마이저 저장 딕셔너리
        for i in range(num_nets):
            self.optimizers[i] = self.init_optimizer(self.nets[i],net_arch[i], lr, lr_linear)#초기화


    # initialize one network
    def init_net(self, net_arch):
        if net_arch == 'bert-base-uncased':
            net = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = self.n_classes)

        elif net_arch == 'microsoft/codebert-base':
            net =  AutoModelForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels = self.n_classes)
                
        elif net_arch == "microsoft/unixcoder-base":
            net =  AutoModelForSequenceClassification.from_pretrained("microsoft/unixcoder-base", num_labels = self.n_classes)
        elif net_arch == "Salesforce/codet5p-110m-embedding":
            net =  AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5p-110m-embedding", trust_remote_code=True,num_labels = self.n_classes)

        net.to(self.device)
        return net
    

    # initialize one optimizer
    def init_optimizer(self, net, net_arch, lr, lr_linear):
        optimizer_net = None
        if net_arch == 'bert-base-uncased':
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('net_arch: ', net_arch, 'lr: ', lr)

        elif net_arch == 'microsoft/codebert-base':
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('net_arch: ', net_arch, 'lr: ', lr)

        elif net_arch == "microsoft/unixcoder-base":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('net_arch: ', net_arch, 'lr: ', lr)

        elif self.net_arch == "Salesforce/codet5p-110m-embedding":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('net_arch: ', net_arch, 'lr: ', lr)
        return optimizer_net
    


    # EMA initialization
    def init_ema(self, ema_momentum):
        self.emas = {} # 지수이동평균(EMA) 객체 저장 딕셔너리
        for i in range(self.num_nets):#신경망 개수만큼 반복
            self.emas[i] = EMA(self.nets[i], ema_momentum)# 각 신경망에 대한 EMA 초기화
            self.emas[i].register() # 원본 모델의 파라미터를 EMA에 등록

    # switch to eval mode with EMA
    def eval_ema(self):
        for i in range(self.num_nets):#
            self.emas[i].apply_shadow() #각 신경망에 대해 모델 파라미터를 EMA 값으로 설정하고, 백업에 현재 파라미터 저장

    # switch to train mode with EMA
    def train_ema(self):
        for i in range(self.num_nets):
            self.emas[i].restore()

    # switch to train mode
    def train(self):
        for i in range(self.num_nets):
            self.nets[i].train() #각 신경망 그룹을 train 시킨다.

    # switch to eval mode
    def eval(self):
        for i in range(self.num_nets):
            self.nets[i].eval()

    # forward one network
    def forward_net(self, net, net_arch, x, y=None):
        logits = 0
        if net_arch == 'bert-base-uncased':
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits

        elif net_arch == 'microsoft/codebert-base':
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True)
            logits = outs.logits if hasattr(outs, 'logits') else outs

        elif net_arch == "microsoft/unixcoder-base":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True)
            logits = outs.logits if hasattr(outs, 'logits') else outs
        
        elif self.net_arch == "Salesforce/codet5p-110m-embedding":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outs.logits if hasattr(outs, 'logits') else outs
            
        return logits


    # forward the group of networks from the same batch input
    def forward(self, x, y=None):
        outs = []
        for i in range(self.num_nets):
            outs.append(self.forward_net(self.nets[i], self.net_arch[i], x, y))
        return outs
    
    # update one network
    def update_net(self, net, optimizer, loss):
        # always clear any previously calculated gradients before performing backward pass
        net.zero_grad()# 3.손실에 대한 그래디언트를 계산합니다
        loss.backward(retain_graph=True)
        optimizer.step()# 4.그래디언트를 사용하여 모델의 가중치를 업데이트합니다

    # update the group of networks
    
    def update(self, losses):
        for i in range(self.num_nets):
            self.update_net(self.nets[i], self.optimizers[i], losses[i])

    # update the group of networks with EMA
    def update_ema(self):
        #netgroup 객체 내의 모든 네트워크들의 EMA가 업데이트 됨
        for i in range(self.num_nets):
            self.emas[i].update()

    # save & load
    # save the group of models
    def save_model(self, path, name, ema_mode=False):
        # use ema model for evaluation
        ema_model = {}
        for i in range(self.num_nets):
            filename = os.path.join(path, name + '_net' + str(i) + '.pth')
            # switch to eval mode with EMA
            self.nets[i].eval()
            if ema_mode:
                    self.emas[i].apply_shadow()
            ema_model[i] = self.nets[i].state_dict()
            # restore training mode
            if ema_mode:
                self.emas[i].restore()
            self.nets[i].train()

            # save model
            torch.save({'model': self.nets[i].state_dict(),
                       'optimizer': self.optimizers[i].state_dict(),
                       'ema_model': ema_model[i]},
                       filename)
        print('Save model to {}'.format(path))




    # load the group of models
    def load_model(self, path, name, ema_mode=False):
        ema_model = {}
        for i in range(self.num_nets):
            filename = os.path.join(path, name + '_net' + str(i) + '.pth')
            checkpoint = torch.load(filename)
            self.nets[i].load_state_dict(checkpoint['model'])
            self.optimizers[i].load_state_dict(checkpoint['optimizer'])
            if ema_mode:
                ema_model[i] = deepcopy(self.nets[i])
                ema_model[i].load_state_dict(checkpoint['ema_model'])
                self.emas[i].load(ema_model[i])
            print('Load model from {}'.format(filename))