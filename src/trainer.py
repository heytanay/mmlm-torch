import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast 

from .utils import wandb_log

class Trainer:
    def __init__(self, config, dataloaders, optimizer, model, loss_fns, scheduler, device="cuda:0", apex=False):
        self.train_loader, self.valid_loader = dataloaders
        self.train_loss_fn, self.valid_loss_fn = loss_fns
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.model = model
        self.device = torch.device(device)
        self.apex = apex

        if self.apex:
            self.scaler = GradScaler()
    
    def _train_batch(self, x, y):
        """
        Trains the model on 1 batch of data
        """
        if self.apex:
            with autocast(enabled=True):
                out = self.model(x)
                loss = self.train_loss_fn(out, y).squeeze()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer=self.optimizer)
                self.scaler.update()
        else:
            out = self.model(x)
            loss = self.train_loss_fn(out, y).squeeze()
            loss.backward()
        self.optimizer.zero_grad()
        return loss

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        
        prog_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for img_emb, text_emb in prog_bar:
            img_emb = self._convert_if_not_tensor(img_emb)
            text_emb = self._convert_if_not_tensor(text_emb)

            if self.apex:
                out = self.model(text_emb)
                loss = self.train_loss_fn(out, img_emb)
                loss.backward()
            

    @torch.no_grad()
    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        pass

    def fit(self, fold: str, epochs: int = 10, output_dir: str = "/kaggle/working/", custom_name: str = 'model.pth'):
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e+7)
        best_preds = None
        for epx in range(epochs):
            print(f"{'='*20} Epoch: {epx+1} / {epochs} {'='*20}")

            train_running_loss = self.train_one_epoch()
            print(f"Training loss: {train_running_loss:.4f}")

            valid_loss, preds = self.valid_one_epoch()
            print(f"Validation loss: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_model(output_dir, custom_name)
                print(f"Saved model with val_loss: {best_loss:.4f}")
                best_preds = preds
            
            # Log
            if Config['wandb']:
                wandb_log(
                    train_loss=train_running_loss,
                    val_loss=valid_loss
                )
        return best_preds
            
    def save_model(self, path, name, verbose=False):
        """
        Saves the model at the provided destination
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            print("Errors encountered while making the output directory")

        torch.save(self.model.state_dict(), os.path.join(path, name))
        if verbose:
            print(f"Model Saved at: {os.path.join(path, name)}")

    def _convert_if_not_tensor(self, x, dtype):
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)