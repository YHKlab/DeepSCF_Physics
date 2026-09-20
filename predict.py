import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from model import DeepSCF
from config_loader import get_args
from utils import log
from utils import siestaio as io
from utils.unit import ang2bohr
import pickle
import h5py
import numpy as np
import glob


def convert(args, device, test_list, model):

    def load_hdf5(path):
        with h5py.File(path, 'r') as f:
            target = f['target'][:]
            feature = f['feature'][:]
        x = torch.from_numpy(feature).float()
        y = torch.from_numpy(target).float()

        return x.unsqueeze(0), y.unsqueeze(0)

    model.eval()

    # loss function
    loss = nn.MSELoss(reduction='sum')

    # get reference mesh data (ang -> bohr)
    step = np.array(args.predict.target.grid_spacing) * ang2bohr

    # logging
    Logger = log.logger(path='./predict.txt')

    # convert predicted mesh data
    with torch.no_grad():
        for path in test_list:
            data, target = load_hdf5(path)
            data, target = data.to(device), target.to(device)
            output = model(data)
            initial = args.model.rho0_std * data[:,args.model.rho0_index,:,:,:] + args.model.rho0_mean

            # logging
            Logger.update(path=path, loss=loss(output, target).item())

            print(f'Path: {path}')
            print(f'Initial loss: {loss(initial, target).item()}')
            print(f'ML loss: {loss(output, target).item()}')

            # Tensor object to Numpy array
            output = output.cpu().numpy()
            name = path.split('/')[-1].split('.h5')[0] + '.RHO'

            # write density in SIESTA format
            mesh = np.array(np.shape(output)[1:])
            cell = np.zeros((3,3))
            cell[0] = np.array([1,0,0]) * mesh[0] * step
            cell[1] = np.array([0,1,0]) * mesh[1] * step
            cell[2] = np.array([0,0,1]) * mesh[2] * step
            io.writeGrid(name, cell, mesh, output)

    # save log
    Logger.save()


def main(args: OmegaConf):

    # gpu/cpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = DeepSCF(args.model).to(device)

    # load model & datalist
    path_model = args.predict.model
    checkpoint = torch.load(path_model, map_location=device)

    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # target path
    path_data = args.predict.target.path

    if os.path.isdir(path_data):
        target_list = glob.glob(path_data + '/*.h5')
    elif path_data.split('.')[-1] == 'h5':
        target_list = [path_data]
    else:
        with open(path_data,'rb') as f:
            data_list = pickle.load(f)

        if args.is_test:
            target_list = data_list['test']
        else:
            target_list = data_list['train']

    # convert to SIESTA RHO data
    convert(args, device, target_list, model)



if __name__=='__main__':
    input_args = get_args()
    main(input_args)
