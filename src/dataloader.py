import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class EmbeddingData(Dataset):
    def __init__(self, image_embeddings, text_embeddings):
        """
        Custom Dataset for loading the image and text embedding data
        """
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

        assert len(self.image_embeddings) == len(text_embeddings)

    def __getitem__(self, idx):
        img_emb = torch.tensor(self.image_embeddings[idx])
        text_emb = torch.tensor(self.text_embeddings[idx])
        
        return img_emb, text_emb
    
    def __len__(self):
        return len(self.image_embeddings)