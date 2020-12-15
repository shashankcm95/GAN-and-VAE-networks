import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torchvision
import numpy as np

def get_dataloader(size_of_batch, pad = False):
    if pad:
        transform = transforms.Compose([
                transforms.Pad(padding = 2, padding_mode = 'edge'),
                transforms.ToTensor()
                ])
    else:
        transform = transforms.ToTensor()
    
    dataset = MNIST(root = './data', train = True, download = True, 
                    transform = transform)
    dataloader = DataLoader(dataset = dataset, batch_size = size_of_batch, 
                            shuffle = True)
    return dataloader

def makedirs(image_dir,checkpoint_dir):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    

def get_optim(model, lr):
    return optim.Adam(model.parameters(), betas = [0.5, 0.999], lr = lr)

def get_system():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def show_process(epoch, step, step_per_epoch, g_log, d_log):
    print('Epoch [{}], Step [{}/{}], Losses: G [{:8f}], D [{:8f}]'.format(
            epoch, step, step_per_epoch, g_log[-1], d_log[-1]))
    return    

def save_model(model, optim, logs, ckpt_dir, filename):
    file_path = os.path.join(ckpt_dir, filename)
    state = {'model': model.state_dict(),
             'optim': optim.state_dict(),
             'logs': tuple(logs),
             'steps': len(logs)}
    torch.save(state, file_path)
    return

def save_image(image, nrow, epoch, step, sample_dir):
    filename = 'epoch_{}_step_{}.png'.format(epoch, step)
    file_path = os.path.join(sample_dir, filename)
    torchvision.utils.save_image(image, file_path, nrow)
    return

def load_model(model, file_path):
    state = torch.load(file_path)
    model.load_state_dict(state['model'])
    return