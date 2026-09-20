from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import pickle
import h5py
import os

def random_split(path, ratio, augmentation, seed):

    # return train and validation data paths
    print(f'{path}/*.h5')
    dataset = sorted(glob.glob(f'{path}/*.h5'))

    # rearrange the data for augmentation
    np.random.seed(seed)
    data_arange = np.arange(len(dataset)/augmentation)

    # random separation
    np.random.seed(seed)
    random_arange = np.random.permutation(data_arange)
    ntrain = round(len(data_arange)*ratio)
    train_arange = random_arange[:ntrain]
    test_arange = random_arange[ntrain:]

    train_list = []
    test_list = []

    # define test and train dataset path lists
    for i in train_arange:
        for j in range(augmentation):
            path = dataset[int(i*augmentation+j)]
            train_list.append(path)
    for i in test_arange:
        for j in range(augmentation):
            path = dataset[int(i*augmentation+j)]
            test_list.append(path)

    return train_list, test_list


def create_dataloader(args):

    # split into train and validation datasets
    path = args.dataset.path
    ratio = args.dataset.ratio
    train_list, test_list = random_split(path,
                                         args.dataset.ratio,
                                         args.dataset.augmentation,
                                         args.dataset.seed
                                        )

    # save dataset information
    dataset_info = {'train': list(train_list),
                 'test': list(test_list)
                }
    with open('data.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)

    # dataset
    train_set = HDF5Dataset(train_list, args.dataset.use_cache, args.dataset.cache_size, args.dataset.mesh_after_crop)
    test_set = HDF5Dataset(
        test_list,
        use_cache=False,
        mesh_after_crop=args.dataset.mesh_after_crop,
    )

    # define dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.dataset.batch_size,
                              shuffle=True,
                              num_workers=args.dataloader.num_workers,
                              pin_memory=args.dataloader.pin_memory
                             )

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=args.dataloader.num_workers,
                             pin_memory=args.dataloader.pin_memory
                           )

    return train_loader, test_loader

def load_dataloader(path, args):

    # load dataset information
    with open(path,'rb') as f:
        data_list = pickle.load(f)

    # dataset
    train_list = data_list['train']
    test_list = data_list['test']
    train_set = HDF5Dataset(train_list, args.dataset.use_cache, args.dataset.cache_size, args.dataset.mesh_after_crop)
    test_set = HDF5Dataset(
        test_list,
        use_cache=False,
        mesh_after_crop=args.dataset.mesh_after_crop,
    )



    # define dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.dataset.batch_size,
                              shuffle=True,
                              num_workers=args.dataloader.num_workers,
                              pin_memory=args.dataloader.pin_memory
                             )

    test_loader = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=True,
                            num_workers=args.dataloader.num_workers,
                            pin_memory=args.dataloader.pin_memory
                           )

    return train_loader, test_loader


class HDF5Dataset(Dataset):

    def __init__(self, path_list, use_cache = False, cache_size = 1, mesh_after_crop = 0):

        super(HDF5Dataset, self).__init__()
        self.path = path_list
        self.use_cache = use_cache
        self.cached_index = []
        self.cached_target = []
        self.cached_feature = []

        self.mesh_after_crop = mesh_after_crop

        if use_cache:
            self.set_cache_data(path_list, cache_size)

    def __getitem__(self, index):

        if index in self.cached_index:
            i = self.cached_index.index(index)
            x = self.cached_feature[i]
            y = self.cached_target[i]
        else:
            path = self.path[index]
            target, feature = self.load_hdf5(path)
            x = torch.from_numpy(feature)
            y = torch.from_numpy(target)

        if self.mesh_after_crop:
            x = CropCent3d(x, self.mesh_after_crop)
            y = CropCent3d(y, self.mesh_after_crop)

        return x.float(), y.float()

    def __len__(self):

        return len(self.path)

    def set_cache_data(self, path_list, cache_size):

        cached_target = []
        cached_feature = []

        # choice the indices for cached data
        index_arange = np.arange(len(path_list))
        if cache_size == 'max':
            cached_index = index_arange
        else:
            cached_index = np.random.permutation(index_arange)[:cache_size]

        for index in cached_index:
            path = path_list[index]
            target, feature = self.load_hdf5(path)
            x = torch.from_numpy(feature)
            y = torch.from_numpy(target)

            # cache the data in memory
            cached_feature.append(x)
            cached_target.append(y)

        self.cached_index = list(cached_index)
        self.cached_target = torch.stack(cached_target)
        self.cached_feature = torch.stack(cached_feature)


    def load_hdf5(self, path):

        f = h5py.File(path,'r')
        target = f['target'][:]
        feature = f['feature'][:]
        f.close()

        return target, feature

def CropCent3d(x, output_meshsize):
    # Perform center cropping
    input_meshsize = x.shape[-1]
    crop_size = (input_meshsize - output_meshsize) // 2
    if x.ndim == 4:
        x = x[:,
            crop_size : (x.shape[-1] - crop_size),
            crop_size : (x.shape[-1] - crop_size),
            crop_size : (x.shape[-1] - crop_size)]
    elif x.ndim ==3 :
        x = x[
            crop_size : (x.shape[-1] - crop_size),
            crop_size : (x.shape[-1] - crop_size),
            crop_size : (x.shape[-1] - crop_size)]

    return x
