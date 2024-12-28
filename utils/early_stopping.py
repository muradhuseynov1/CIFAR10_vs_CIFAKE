import torch
import numpy as np

class EarlyStopping:
    #stops training if validation loss doesn't improve for "patience" number of epochs
    #default patience is 10 epochs
    def __init__(self, patience=10, checkpoint_path='best_model.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    #it saves the best model checkpoint, it's saved as pth file
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

"""
for our short trainings we never reached limit of patience
however we are aware that early stoppping function is important for longer trainings
"""