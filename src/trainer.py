import os
import gc
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast 

from .utils import wandb_log
from .config import Config

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
    
    def _batch_calculation(self, x, y, loss_fn, mode="train", optimizer=None):
        """
        Trains / Validates the model on 1 batch of data
        """
        if mode == "train":
            if self.apex:
                with autocast(enabled=True):
                    out = self.model(x)
                    loss = loss_fn(out, y).squeeze()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer=self.optimizer)
                    self.scaler.update()
            else:
                out = self.model(x)
                loss = loss_fn(out, y).squeeze()
                loss.backward()
                self.optimizer.step()
            self.optimizer.zero_grad()

            return loss
        
        elif mode == "valid":
            with torch.no_grad():
                out = self.model(x)
                loss = loss_fn(out, y).squeeze()
            return loss
        
        else:
            assert True is False, "Can't have a mode other than 'train' or 'valid'"

    def train_one_epoch(self):
        """
        Trains the model for 1 epoch
        """
        self.model.train()
        running_loss = 0
        prog_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for idx, (img_emb, text_emb) in prog_bar:
            img_emb = self._convert_if_not_tensor(img_emb)
            text_emb = self._convert_if_not_tensor(text_emb)
            
            loss = self._batch_calculation(
                text_emb, img_emb, 
                self.train_loss_fn, 
                "train", 
                self.optimizer
            )
            loss_itm = loss.item()
            prog_bar.set_description(f"loss: {loss_itm}")
            running_loss += loss_itm
        running_loss = running_loss / len(self.train_loader)

        # Tidy 🧹
        del img_emb, text_emb, loss
        torch.cuda.empty_cache()
        gc.collect()

        return running_loss

    def valid_one_epoch(self):
        """
        Validates the model for 1 epoch
        """
        self.model.eval()
        running_loss = 0
        prog_bar = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader))
        for idx, (img_emb, text_emb) in prog_bar:
            img_emb = self._convert_if_not_tensor(img_emb)
            text_emb = self._convert_if_not_tensor(text_emb)

            loss = self._batch_calculation(
                text_emb, 
                img_emb,
                self.valid_loss_fn,
                "valid",
                None
            )
            loss_itm = loss.item()
            prog_bar.set_description(f"val_loss: {loss_itm}")
            running_loss += loss_itm

        running_loss = running_loss / len(self.valid_loader)
        
        # Tidy 🧹
        del img_emb, text_emb, loss
        torch.cuda.empty_cache()
        gc.collect()

        return running_loss

    def fit(self, epochs: int = 10, custom_name: str = 'model.pth'):
        """
        Low-effort alternative for doing the complete training and validation process
        """
        best_loss = int(1e+7)
        best_preds = None
        for epx in range(epochs):
            print(f"{'='*20} Epoch: {epx+1} / {epochs} {'='*20}")

            train_running_loss = self.train_one_epoch()
            print(f"Training loss: {train_running_loss:.4f}")

            valid_loss = self.valid_one_epoch()
            print(f"Validation loss: {valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                self.save_model(custom_name)
                print(f"Saved model with val_loss: {best_loss:.4f} at ./{custom_name}")
            
            # Log
            if Config.wandb:
                wandb_log(
                    train_loss=train_running_loss,
                    val_loss=valid_loss
                )
        return best_preds
            
    def save_model(self, name):
        """
        Saves the model
        """
        torch.save(self.model.state_dict(), os.path.join("./", name))

    def _convert_if_not_tensor(self, x, dtype):
        if self._tensor_check(x):
            return x.to(self.device, dtype=dtype)
        else:
            return torch.tensor(x, dtype=dtype, device=self.device)

    def _tensor_check(self, x):
        return isinstance(x, torch.Tensor)