import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoModel
from utils.ema import EMA
from models.model import TextClassifier

from transformers import PreTrainedModel, AutoModel



class CustomModel(nn.Module):
    def __init__(self, transformer_model_name, n_classes=7):
        super(CustomModel, self).__init__()

        # Load the transformer model without the classification head
        config = AutoConfig.from_pretrained(transformer_model_name)
        print("############")
        print(config)
        self.text_transformer = AutoModel.from_pretrained(transformer_model_name)

        # Use the same dimensions as the transformer model for nn.Embedding
        n_dims = config.vocab_size
        print("############")
        print(n_dims)
        n_factors = config.hidden_size
        print("############")
        print(n_factors)
        # Modify the linear layer to match the number of classes
        self.linear_layer = nn.Linear(self.text_transformer.config.hidden_size + n_factors, n_classes)





model = CustomModel("Salesforce/codet5p-110m-embedding")
net = model