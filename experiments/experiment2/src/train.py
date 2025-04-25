import time

import torch
from torch import nn
from torch.autograd import Variable

from utils import MinNormSolver, get_device


def loss_func(y_pred, y_true):
    loss_fn = nn.L1Loss()
    loss = loss_fn(y_pred, y_true)
    return loss


def train(mul_adj, phy_adj, net_adj, model=None, save_path='', config={}, train_dataloader=None):
    phy_params = []
    net_params = []
    multi_params = []

    for index, (name, param) in zip(enumerate(model.parameters()), model.named_parameters()):
        if name.startswith('phy'):
            phy_params.append(param)
        elif name.startswith('net'):
            net_params.append(param)
        else:
            multi_params.append(param)

    net_optimizer = torch.optim.SGD(net_params, lr=0.001, weight_decay=config['decay'], momentum=0.9)
    phy_optimizer = torch.optim.SGD(phy_params, lr=0.00001, weight_decay=config['decay'], momentum=0.9)
    optimizer = torch.optim.SGD(multi_params, lr=0.001, weight_decay=config['decay'], momentum=0.9)

    train_loss_list = []

    device = get_device()

    min_loss = 1e+8

    i = 0
    epoch = config['epoch']

    model.train()

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, py, ny, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, py, ny, edge_index = [item.float().to(device) for item in [x, py, ny, edge_index]]

            phy_optimizer.zero_grad()
            net_optimizer.zero_grad()

            outs, learned_graph, phy_graph, net_graph = model(x, mul_adj, phy_adj, net_adj)

            phy_out = outs[0].to(device)
            net_out = outs[1].to(device)

            phy_loss = loss_func(phy_out, py)

            phy_loss = phy_loss.requires_grad_()
            phy_loss.backward(retain_graph=True)

            net_loss = loss_func(net_out, ny)
            y_loss = net_loss.cpu()
            loss_ = str(y_loss.data.numpy())
            with open('./loss.txt', 'a') as f:
                f.write(',')
                f.write(loss_)
            net_loss = net_loss.requires_grad_()
            net_loss.backward(retain_graph=True)

            grads = [[]]
            new_list = []
            for param in phy_params:
                if param.grad is not None:
                    grads[0].append(Variable(param.grad.data.clone(), requires_grad=False))
            for param in net_params:
                if param.grad is not None:
                    new_list.append(Variable(param.grad.data.clone(), requires_grad=False))
            grads.append(new_list)

            scale = []
            sol, min_norm = MinNormSolver.find_min_norm_element()
            scale.append(float(sol[0]))
            scale.append(float(sol[1]))

            optimizer.zero_grad()

            loss = scale[0] * phy_loss + scale[1] * net_loss

            for param in multi_params:
                param.requires_grad = False
            loss.backward()
            phy_optimizer.step()
            net_optimizer.step()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss = acu_loss + loss.item()

            i = i + 1

        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss), flush=True
        )

        if acu_loss < min_loss:
            torch.save(model.state_dict(), save_path)
            min_loss = acu_loss

    return train_loss_list, learned_graph
