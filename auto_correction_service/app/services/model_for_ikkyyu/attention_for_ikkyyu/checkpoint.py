import torch

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_accuracy": [],
    "validation_losses": [],
    "validation_accuracy": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
}

def load_checkpoint(path, gpu_index, cuda=use_cuda):
    if cuda:
        # print(path)
        # return torch.load(path)
        return torch.load(path, map_location='cuda:{}'.format(str(gpu_index)))
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)