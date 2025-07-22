import torch
from torch.utils.tensorboard import SummaryWriter

def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)

def load_model(model, path='model.pth'):
    model.load_state_dict(torch.load(path))

def get_writer(log_dir='runs/exp1'):
    return SummaryWriter(log_dir=log_dir)
