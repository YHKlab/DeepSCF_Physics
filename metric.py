import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.convolution import gradient_3point, gradient_5point, laplacian_3point, laplacian_5point
from utils import log, data
import os


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    # loss function
    loss_function = nn.MSELoss(reduction=args.train.loss.reduction)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)

        # loss function
        loss = loss_function(output, target)

        # step
        loss.backward()
        optimizer.step()

        # evaluate error
        train_loss += loss.item()

        if batch_idx % args.logger.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss  /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f} \n'.format(train_loss))

    return train_loss


def train_physics(args, model, device, train_loader, optimizer, epoch):

    model.train()

    # loss function
    loss_function = nn.MSELoss(reduction=args.train.loss.reduction)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # regression loss
        loss = loss_function(output, target)

        # loss function
        if args.train.loss.regularization != None:
            total_loss = loss

            if 'gradient_3p' in args.train.loss.regularization:

                alpha = args.train.loss.regularization_weight['gradient']

                # gradient target
                Gx_target, Gy_target, Gz_target = gradient_3point(target, device)
                Gx_output, Gy_output, Gz_output = gradient_3point(output, device)

                total_loss += alpha * loss_function(Gx_output, Gx_target)
                total_loss += alpha * loss_function(Gy_output, Gy_target)
                total_loss += alpha * loss_function(Gz_output, Gz_target)

            if 'gradient_5p' in args.train.loss.regularization:

                alpha = args.train.loss.regularization_weight['gradient']

                # gradient target
                Gx_target, Gy_target, Gz_target = gradient_5point(target, device)
                Gx_output, Gy_output, Gz_output = gradient_5point(output, device)

                total_loss += alpha * loss_function(Gx_output, Gx_target)
                total_loss += alpha * loss_function(Gy_output, Gy_target)
                total_loss += alpha * loss_function(Gz_output, Gz_target)

            if 'laplacian_3p' in args.train.loss.regularization:

                alpha = args.train.loss.regularization_weight['laplacian']

                # gradient target
                Lap_target = laplacian_3point(target, device)
                Lap_output = laplacian_3point(output, device)

                total_loss += alpha * loss_function(Lap_output, Lap_target)

            if 'laplacian_5p' in args.train.loss.regularization:

                alpha = args.train.loss.regularization_weight['laplacian']

                # gradient target
                Lap_target = laplacian_5point(target, device)
                Lap_output = laplacian_5point(output, device)

                total_loss += alpha * loss_function(Lap_output, Lap_target)

        else:
            total_loss = loss

        # step
        total_loss.backward()
        optimizer.step()

        # evaluate error
        train_loss += loss.item()

        if batch_idx % args.logger.interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss  /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f} \n'.format(train_loss))

    return train_loss


def test(args, model, device, test_loader):

    model.eval()

    # loss
    test_loss = 0
    ne = 0
    mae = 0

    # loss functions
    mse_loss = nn.MSELoss(reduction=args.train.loss.reduction)
    mae_loss = nn.L1Loss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # mse loss
            test_loss += mse_loss(output, target).item()

            # percentage loss
            mae += mae_loss(output, target).item()
            ne += torch.sum(target).item()

    test_loss  /= len(test_loader.dataset)
    accuracy = (1-mae/ne)*100

    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    print('        : Accuracy: {:.4f} %\n'.format(accuracy))

    return test_loss, accuracy


def interpol(grid, target):
    # target: [B, C, D, H, W] or [B, D, H, W]

    if target.dim() == 4:  # [B, D, H, W] >>> [B, 1, D, H, W]
        target = target.unsqueeze(1)

    target = F.interpolate(target, size=grid, mode="trilinear", align_corners=False)

    return target.squeeze(1)  # [B, D, H, W]
