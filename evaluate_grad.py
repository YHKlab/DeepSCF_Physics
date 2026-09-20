import torch
import torch.nn as nn
import os
from omegaconf import OmegaConf
from model import DeepSCF
from utils import data, log
import pickle
import h5py
import glob
from config_loader import get_args
from model.convolution import gradient_3point


def evaluate(args, device, test_list, model, train_list = None):

    def load_hdf5(path):
        with h5py.File(path, 'r') as f:
            target = f['target'][:]
            feature = f['feature'][:]
        x = torch.from_numpy(feature).float()
        y = torch.from_numpy(target).float()

        return x.unsqueeze(0), y.unsqueeze(0)

    model.eval()


    # logging
    Logger = log.logger(path='./evaluate.txt')
    summary = open('summary_grad.txt', 'w')

    # evaluate test loss

    # loss function
    loss = nn.L1Loss(reduction='sum')
    gradient_mae = 0
    tot_grad = 0 # total absolute gradient

    ### evaluate test loss
    if args.is_test:
        with torch.no_grad(): # no train
            for path in test_list:
                print(path)
                # load output within GPU
                data, target = load_hdf5(path)
                data, target = data.to(device), target.to(device)
                output = model(data)

                # apply gradient kernel
                output_grad_x, output_grad_y, output_grad_z, = gradient_3point(output, device)
                target_grad_x, target_grad_y, target_grad_z, = gradient_3point(target, device)

                # calculate gradient
                output_grad = torch.sqrt(output_grad_x**2 + output_grad_y**2 + output_grad_z**2)
                target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + target_grad_z**2)

                gradient_mae += loss(output_grad, target_grad).item()
                tot_grad += torch.sum(target_grad).item()

                # percentage error
                p_error = 100 * loss(output_grad, target_grad).item() / torch.sum(target_grad).item()

                # logging
                Logger.update(path=path, loss=p_error)

        # total percentage error
        percentage_error = gradient_mae/tot_grad*100

        summary.write(f'Test gradient percentage error: {percentage_error} \n')
        print(f'Percentage error: {percentage_error} \n')

    ### evaluate train loss
    else:
        if train_list != None:
            with torch.no_grad():
                for path in train_list:

                    print(path)
                    # load output within GPU
                    data, target = load_hdf5(path)
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    # apply gradient kernel
                    output_grad_x, output_grad_y, output_grad_z, = gradient_3point(output, device)
                    target_grad_x, target_grad_y, target_grad_z, = gradient_3point(target, device)

                    # calculate gradient
                    output_grad = torch.sqrt(output_grad_x**2 + output_grad_y**2 + output_grad_z**2)
                    target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + target_grad_z**2)

                    gradient_mae += loss(output_grad, target_grad).item()
                    tot_grad += torch.sum(target_grad).item()

                    # percentage error
                    p_error = 100 * loss(output_grad, target_grad).item() / torch.sum(target_grad).item()

                    # logging
                    Logger.update(path=path, loss=p_error)


            # total percentage error
            percentage_error = gradient_mae/tot_grad*100
            summary.write(f'Train gradient percentage error: {percentage_error} \n')
            print(f'Percentage error: {percentage_error} \n')

    # save log
    Logger.save()
    summary.close()

def main(args: OmegaConf):

    # gpu/cpu
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = DeepSCF(args.model).to(device)

    # load model & datalist
    path_model = args.evaluate.model
    checkpoint = torch.load(path_model, map_location=device)

    if type(checkpoint) == dict:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # target path
    if args.evaluate.target:
        path_data = args.evaluate.target
        if os.path.isdir(path_data):
            target_list = glob.glob(path_data + '/*.h5')
            evaluate(args, device, target_list, model)

        elif path_data.split('.')[-1] == 'h5':
            target_list = [path_data]
            evaluate(args, device, target_list, model)

        else:
            with open(path_data,'rb') as f:
                data_list = pickle.load(f)
            target_list = data_list['test']
            train_list = data_list['train']
            evaluate(args, device, target_list, model, train_list)
    else:
        # create the dataloader
        train_loader, test_loader = data.create_dataloader(args)
        with open('data.pkl', 'rb') as f:
            data_list = pickle.load(f)
        target_list = data_list['test']
        train_list = data_list['train']
        evaluate(args, device, target_list, model, train_list)



if __name__=='__main__':
    input_args = get_args()
    main(input_args)
