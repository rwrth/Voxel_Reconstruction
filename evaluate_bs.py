import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model_bs import MyLoss
import torch
import gc
import numpy as np


def validate_loss(vali_loader, model, G,graph_vox, batch_size_v, save, writer, loss_f, voxel_pos):
    dataiter = iter(vali_loader)
    input, label = dataiter.next()
    #inp = input.to(dtype=torch.double)#torch.reshape(input, (600,50)).to(dtype=torch.double)
    prediction = model(G, graph_vox, input ,batch_size_v, voxel_pos)
    # The loss is computed only for labeled nodes.
    validation_loss = loss_f(prediction.double(),label, voxel_pos[600:])
    if save:
        writer.add_scalar("validation_loss", validation_loss.item())
        writer.flush()
    del input, label, prediction, dataiter
    gc.collect()
    return validation_loss


def performance(G, graph_vox, vali_loader, model, voxel_pos,bs):
    dif = 0
    mean = []
    std = []
    r_voll = []
    r_leer = []
    for input, labels in vali_loader:
        inp = input.to(dtype=torch.double)
        prediction = model(G, graph_vox, inp.double(), bs, voxel_pos)
        dif_sum = torch.div(torch.abs(torch.sum(prediction)-torch.sum(labels)),torch.sum(labels, axis = 1))
        dif = np.append(dif, dif_sum.detach().numpy())
        mean.append(torch.mean(prediction).detach().numpy())
        std.append(torch.std(prediction).detach().numpy())
        predT = (prediction.detach().numpy().reshape(bs,8000) >= 1)
        truT = (labels.detach().numpy().reshape(bs,8000) >= 1)
        r_voll.append(np.sum((predT & truT), axis = 1)/np.sum(truT, axis = 1))
        r_leer.append(np.sum((np.invert(predT) & np.invert(truT)),axis = 1)/np.sum(np.invert(truT), axis =1))
    return np.mean(dif[1:]), np.std(dif[1:]), np.mean(r_voll), np.std(r_voll), np.mean(r_leer), np.std(r_leer)
