import nltk
import warnings
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from nltk.corpus import wordnet as wn
from pytorch_pretrained_biggan.utils import IMAGENET
from transformers import AutoTokenizer, AutoModel
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample, one_hot_from_names

import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

from src.trainer import Trainer
from src.models import *
from src.dataloader import *
from src.utils import *
from src.config import *

nltk.download('wordnet')
warnings.simplefilter('ignore')

if __name__ == "__main__":
    # Load the saved embeddings
    img_emb = np.load("working/image_embeddings_large.npy", allow_pickle=True)
    text_emb = np.load("working/text_embeddings_large.npy", allow_pickle=True)

    # Convert internal tensors to Torch tensors again
    img_embeddings = [torch.from_numpy(np_arr) for np_arr in img_emb]
    text_embeddings = [torch.from_numpy(np_arr) for np_arr in text_emb]

    # Split the data and make dataloaders
    split_pcent = 0.85
    split_nb = int(len(img_embeddings) * split_pcent)

    train_ie, train_te = img_embeddings[:split_nb], text_embeddings[:split_nb]
    valid_ie, valid_te = img_embeddings[split_nb:], text_embeddings[split_nb:]

    train_data = EmbeddingData(train_ie, train_te)
    valid_data = EmbeddingData(valid_ie, valid_te)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=1, 
        shuffle=True, 
        num_workers=2
    )

    valid_loader = DataLoader(
        valid_data, 
        batch_size=1,
        shuffle=True,
        num_workers=2
    )

    # Define model, loss functions and optimizers
    model = MapperModel()
    model = model.to(torch.device('cuda'))
    train_loss_fn, valid_loss_fn = nn.L1Loss(), nn.L1Loss()
    optim = torch.optim.Adam(model.parameters())

    # Define a Trainer and do the fitting
    trainer = Trainer(
        model=model,
        config=Config, 
        dataloaders=(train_loader, valid_loader),
        optimizer=optim,
        loss_fns=(train_loss_fn, valid_loss_fn),
        scheduler=None,
        apex=True
    )

    trainer.fit()