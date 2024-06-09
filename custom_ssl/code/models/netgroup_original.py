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
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM , AutoModelForCausalLM
from utils.ema import EMA
from models.model import TextClassifier
import pdb
from transformers import PreTrainedModel, AutoModel, BertTokenizer, AutoTokenizer

# 추가
class CustomModel(nn.Module): #codet5p 용
    def __init__(self, transformer_model_name, n_classes=7):
        super(CustomModel, self).__init__()

        # Load the transformer model without the classification head
        config = AutoConfig.from_pretrained(transformer_model_name,  trust_remote_code=True)
        self.text_transformer = AutoModel.from_pretrained(transformer_model_name,  trust_remote_code=True)

        # Linear layer for classification
        self.linear_layer = nn.Linear(config.embed_dim, n_classes)


    def forward(self, input_ids, attention_mask=None, labels=None):  # Add 'labels' argument

        # Obtain transformer output
        # transformer_output = self.text_transformer(input_ids, attention_mask=attention_mask, labels=labels).last_hidden_state.mean(dim=1)
        transformer_output = self.text_transformer(input_ids, attention_mask=attention_mask)
        # Pass through the linear layer for classification
        logits = self.linear_layer(transformer_output)

        return logits
class CustomModel_2(nn.Module): #graphcodebert 용
    def __init__(self, transformer_model_name, n_classes=7):
        super(CustomModel_2, self).__init__()

        # Load the transformer model without the classification head
        config = AutoConfig.from_pretrained(transformer_model_name,  trust_remote_code=True)
        self.text_transformer = AutoModel.from_pretrained(transformer_model_name,  trust_remote_code=True)

        # Linear layer for classification
        self.linear_layer = nn.Linear(config.hidden_size, n_classes)


    def forward(self, input_ids, attention_mask=None, labels=None):  
        # Obtain transformer output
        
        transformer_output = self.text_transformer(input_ids, attention_mask=attention_mask).last_hidden_state

        # Pass through the linear layer for classification
        
        logits = self.linear_layer(transformer_output[:, 0, :])  # Take the first token's hidden state

        return logits

class NetGroup(nn.Module):
    def __init__(self, net_arch, num_nets, n_classes, device, lr, lr_linear=1e-3):
        super(NetGroup, self).__init__()
        # parameters
        self.net_arch = net_arch
        self.num_nets = num_nets
        self.n_classes = n_classes
        self.device = device
        self.lr = lr

        # initialize the group of networks
        self.nets = {}
        for i in range(num_nets):
            self.nets[i] = self.init_net(net_arch[i])

        # initialize optimizers for the group of networks
        self.optimizers = {}
        for i in range(num_nets):
            # 이부분 수정
            self.optimizers[i] = self.init_optimizer(self.net_arch[i],self.nets[i], lr[i], lr_linear) # 추가 수정

    # initialize one network
    def init_net(self, net_arch):
        # if net_arch == 'bert-base-uncased':
        #     net = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = self.n_classes)
        # elif net_arch == 'bert-base-uncased-2':
        #     net = TextClassifier(num_labels = self.n_classes)
        # net.to(self.device)
        
        # 추가 코드
        if net_arch == 'microsoft/codebert-base':
            net =  AutoModelForSequenceClassification.from_pretrained(net_arch, num_labels = self.n_classes)
            
        elif net_arch == "Salesforce/codet5p-110m-embedding":
            net = CustomModel(net_arch, self.n_classes)

        elif net_arch == "microsoft/unixcoder-base":
            net =  AutoModelForSequenceClassification.from_pretrained(net_arch, num_labels = self.n_classes)
        
        elif net_arch == "microsoft/graphcodebert-base":
            net = CustomModel_2("microsoft/graphcodebert-base", self.n_classes)
        
        elif net_arch == "codesage/codesage-base":
            net = AutoModelForSequenceClassification.from_pretrained(net_arch, trust_remote_code=True,  num_labels = self.n_classes)
            
        # CodeLLama, Starcoder, Deepseekcoder  추가 4/24
        elif net_arch == "codellama/CodeLlama-7b-hf":
            net = AutoModelForSequenceClassification.from_pretrained(net_arch, num_labels = self.n_classes)
        
        elif net_arch == "bigcode/starcoder":
            #net = AutoModelForCausalLM.from_pretrained(net_arch, num_labels = self.n_classes)
            net = AutoModelForSequenceClassification.from_pretrained(net_arch, num_labels = self.n_classes)
            
        elif net_arch == "deepseek-ai/deepseek-coder-6.7b-base":
            #net = AutoModelForCausalLM.from_pretrained(net_arch, trust_remote_code=True, num_labels = self.n_classes)
            net = AutoModelForSequenceClassification.from_pretrained(net_arch, num_labels = self.n_classes)

        net.to(self.device)
        return net
        
        
        
    # initialize one optimizer
    def init_optimizer(self, net_arch, net, lr, lr_linear):
        # if self.net_arch == 'bert-base-uncased':
        #     optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
        #     print('net_arch: ', self.net_arch, 'lr: ', lr)
        # elif self.net_arch == 'bert-base-uncased-2':
        #     optimizer_net = AdamW([{"params": net.bert.parameters(), "lr": lr},
        #                            {"params": net.linear.parameters(), "lr": lr_linear}])
        #     print('net_arch: ', self.net_arch, 'lr: ', lr, 'lr_linear: ', lr_linear)  
        
        
        # 추가코드
        if net_arch == 'microsoft/codebert-base':
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')   

        elif net_arch =="Salesforce/codet5p-110m-embedding":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')  

        elif net_arch == "microsoft/unixcoder-base":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')  
                     
        elif net_arch == "microsoft/graphcodebert-base":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')  
            
        elif net_arch == "codesage/codesage-base":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')  
                
        # CodeLLama, Starcoder, Deepseekcoder  추가 4/24                
        elif net_arch == "codellama/CodeLlama-7b-hf":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')
            
        elif net_arch == "bigcode/starcoder":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')               

        elif net_arch == "deepseek-ai/deepseek-coder-6.7b-base":
            optimizer_net = AdamW(net.parameters(), lr = lr, eps = 1e-8)
            print('\nnet_arch: ', net_arch, '\nlr: ', lr, '\nlr_linear: ', lr_linear,'\n')                
            
        return optimizer_net
    
    # EMA initialization
    def init_ema(self, ema_momentum):
        self.emas = {}
        for i in range(self.num_nets):
            self.emas[i] = EMA(self.nets[i], ema_momentum)
            self.emas[i].register()

    # switch to eval mode with EMA
    def eval_ema(self):
        for i in range(self.num_nets):
            self.emas[i].apply_shadow()

    # switch to train mode with EMA
    def train_ema(self):
        for i in range(self.num_nets):
            self.emas[i].restore()

    # switch to train mode
    def train(self):
        for i in range(self.num_nets):
            self.nets[i].train()

    # switch to eval mode
    def eval(self):
        for i in range(self.num_nets):
            self.nets[i].eval()

    # forward one network
    def forward_net(self, net_arch, net, x, y=None): # 추가 수정
        # if self.net_arch == 'bert-base-uncased':
        #     input_ids = x['input_ids'].to(self.device)
        #     attention_mask = x['attention_mask'].to(self.device)
        #     outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
        # elif self.net_arch == 'bert-base-uncased-2':
        #     x.to(self.device)
        #     outs = net(x)
        
        # 추가 코드
        if net_arch == 'microsoft/codebert-base':
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            # outs = net(input_ids, attention_mask=attention_mask, return_dict=True).last_hidden_state
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits

        elif net_arch == "Salesforce/codet5p-110m-embedding":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y)

        elif net_arch == "microsoft/unixcoder-base":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
          
        elif net_arch == "microsoft/graphcodebert-base":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y)
  
        elif net_arch == "codesage/codesage-base":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
 
        # CodeLLama, Starcoder, Deepseekcoder  추가 4/24
        elif net_arch == "codellama/CodeLlama-7b-hf":    
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
            
        elif net_arch == "bigcode/starcoder":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits
         
        elif net_arch == "deepseek-ai/deepseek-coder-6.7b-base":
            input_ids = x['input_ids'].to(self.device)
            attention_mask = x['attention_mask'].to(self.device)
            outs = net(input_ids, attention_mask=attention_mask, labels=y, return_dict=True).logits    

        return outs
        
        
  
    
    # forward the group of networks from the same batch input
    def forward(self, x, y=None):
        outs = []
        for i in range(self.num_nets):
            outs.append(self.forward_net(self.net_arch[i], self.nets[i], x, y))
        return outs
    
    # 추가 수정함.
    # update one network
    def update_net(self, net, optimizer, loss):
        # always clear any previously calculated gradients before performing backward pass
        #net.zero_grad()
        
        # 추가
        optimizer.zero_grad()
        loss.requires_grad_(True) 
        
        loss.backward(retain_graph=True)
        optimizer.step()

    # update the group of networks
    def update(self, losses):
        for i in range(self.num_nets):
            self.update_net(self.nets[i], self.optimizers[i], losses[i])

    # update the group of networks with EMA
    def update_ema(self):
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