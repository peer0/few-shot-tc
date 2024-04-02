import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertLayer, BertEmbeddings, BertPooler


class TextClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))
    
    #순전파(forward pass) 정의: 모델이 입력에 대해 예측을 만들어내는 단계
    def forward(self, inputs):
        outputs = self.bert(**inputs) #input으로 받은 text데이터를 bert모델에 주입
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)#출력을 평균하여last_hidden_state를 얻은 후, 
        predict = self.linear(pooled_output) #이를 선형 레이어를 통해 예측값을 반환
        return predict 
