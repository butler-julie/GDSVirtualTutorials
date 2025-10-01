import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from torch.utils.data import DataLoader
import sys,os
from tqdm import tqdm

class PFINDataset(Dataset):
    def __init__(self, file_path):
        f = h5py.File(file_path, 'r')
        self.data = torch.from_numpy(f["particles"][:]).float()
        self.masks = torch.from_numpy(f["masks"][:]).float()
        #self.latents = torch.from_numpy(f["latents"][:]).float()
        self.labels = torch.from_numpy(f["labels"][:]).float()
        #self.preds = torch.from_numpy(f["preds"][:]).float()
        self.aug_data = torch.from_numpy(f["aug_data"][:]).float()
        #self.taus = torch.from_numpy(f["taus"][:]).float()
        
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return self.data[idx], self.masks[idx], self.aug_data[idx], self.labels[idx]

class Data(object):
    """Class providing an interface to the input training data. Derived classes should implement the load_data function.
    Attributes:
      file_names: list of data files to use for training
      batch_size: size of training batches
    """

    def __init__(self, batch_size):
        """Stores the batch size and the names of the data files to be read.
        Params:
          batch_size: batch size for training
        """
        self.batch_size = batch_size


    def set_file_names(self, file_names):
        # hook to copy data in /dev/shm
        self.file_names = file_names


    def generate_data(self, shuffle=False):
        """Yields batches of training data until none are left."""
        leftovers = None
        file_names = self.file_names.copy()
        if shuffle:
            np.random.shuffle(file_names)
          
        for cur_file_name in file_names:
            data, mask, aug_data, labels = self.load_data(cur_file_name, shuffle=shuffle)
            # concatenate any leftover data from the previous file
            if leftovers is not None:
                data = self.concat_data(leftovers[0], data)
                mask = self.concat_data(leftovers[1], mask)
                aug_data = self.concat_data(leftovers[2], aug_data)
                labels = self.concat_data(leftovers[3], labels)
                leftovers = None
            num_in_file = self.get_num_samples(data)

            for cur_pos in range(0, num_in_file, self.batch_size):
                next_pos = cur_pos + self.batch_size
                if next_pos <= num_in_file:
                        yield (
                            torch.from_numpy(self.get_batch(data, cur_pos, next_pos)).float(),
                            torch.from_numpy(self.get_batch(mask, cur_pos, next_pos)).float(),
                            torch.from_numpy(self.get_batch(aug_data, cur_pos, next_pos)).float(),
                            torch.from_numpy(self.get_batch(labels, cur_pos, next_pos)).float(),
                        )
                else:
                    leftovers = (
                            self.get_batch(data, cur_pos, num_in_file),
                            self.get_batch(mask, cur_pos, num_in_file),
                            self.get_batch(aug_data, cur_pos, num_in_file),
                            self.get_batch(labels, cur_pos, num_in_file),
                        )

    def count_data(self):
        """Counts the number of data points across all files"""
        num_data = 0
        for cur_file_name in self.file_names:
            data, _, _, _ = self.load_data(cur_file_name)
            num_data += self.get_num_samples(data)
        return num_data

    def is_numpy_array(self, data):
        return isinstance(data, np.ndarray)

    def get_batch(self, data, start_pos, end_pos):
        """Input: a numpy array or list of numpy arrays.
        Gets elements between start_pos and end_pos in each array"""
        if self.is_numpy_array(data):
            return data[start_pos:end_pos]
        else:
            return [arr[start_pos:end_pos] for arr in data]

    def concat_data(self, data1, data2):
        """Input: data1 as numpy array or list of numpy arrays.  data2 in the same format.
        Returns: numpy array or list of arrays, in which each array in data1 has been
          concatenated with the corresponding array in data2"""
        if self.is_numpy_array(data1):
            return np.concatenate((data1, data2))
        else:
            return [self.concat_data(d1, d2) for d1, d2 in zip(data1, data2)]

    def get_num_samples(self, data):
        """Input: dataset consisting of a numpy array or list of numpy arrays.
        Output: number of samples in the dataset"""
        if self.is_numpy_array(data):
            return len(data)
        else:
            return len(data[0])

    def load_data(self, in_file):
        """Input: name of file from which the data should be loaded
        Returns: tuple (X,Y) where X and Y are numpy arrays containing features
            and labels, respectively, for all data in the file
        Not implemented in base class; derived classes should implement this function"""
        raise NotImplementedError


class JetClassData(Data):
    """Loads data stored in hdf5 files
    Attributes:
      features_name, labels_name, spectators_name: names of the datasets containing the features,
      labels, and spectators respectively
    """

    def __init__(
        self,
        batch_size):
        super(JetClassData, self).__init__(batch_size)

    def load_data(self, in_file_name, shuffle = False):
        """Loads numpy arrays from H5 file.
        If the features/labels groups contain more than one dataset,
        we load them all, alphabetically by key."""
        h5_file = h5py.File(in_file_name, "r")
        d = self.load_hdf5_data(h5_file["particles"])
        m = self.load_hdf5_data(h5_file["masks"])
        a = self.load_hdf5_data(h5_file["aug_data"])
        l = self.load_hdf5_data(h5_file["labels"])
        h5_file.close()
        if shuffle:
            idx = np.arange(0, len(d))
            np.random.shuffle(idx)
            d = d[idx]
            m = m[idx]
            a = a[idx]
            l = l[idx]
        return d,m,a,l

    def load_hdf5_data(self, data):
        """Returns a numpy array or (possibly nested) list of numpy arrays
        corresponding to the group structure of the input HDF5 data.
        If a group has more than one key, we give its datasets alphabetically by key"""
        if hasattr(data, "keys"):
            out = [self.load_hdf5_data(data[key]) for key in sorted(data.keys())]
        else:
            out = data[:]
        return out

    def count_data(self):
        """This is faster than using the parent count_data
        because the datasets do not have to be loaded
        as numpy arrays"""
        num_data = 0
        for in_file_name in self.file_names:
            h5_file = h5py.File(in_file_name, "r")
            X = h5_file["particles"]
            if hasattr(X, "keys"):
                num_data += len(X[list(X.keys())[0]])
            else:
                num_data += len(X)
            h5_file.close()
        return num_data

    def __len__(self):
        return self.count_data()

    
if __name__ == "__main__":
    # mydataset = PFINDataset("../datasets/jetnet/processed/test.h5")
    # print(len(mydataset))
    # trainloader = DataLoader(mydataset, batch_size=500, shuffle=False, num_workers=40, pin_memory=True, persistent_workers=True)
    # for i,(d, m, a, l) in enumerate(trainloader):
    #     print(d.shape)
    #     print(m.shape)
    #     print(a.shape)
    #     print(l.shape)
    #     print(l[:5])
    #     print(np.argmax(l[:5].cpu().numpy(), 1))
    #     print(torch.tensor(np.isin(np.argmax(l[:5].cpu().numpy(), 1), [2], invert=True)))
    #     break
    # del mydataset, trainloader
    
    mydataset = JetClassData(batch_size = 5000)
    mydataset.set_file_names(file_names = ["../datasets/jetclass/processed/val_0.h5", "../datasets/jetclass/processed/val_4.h5"])
    print(mydataset.count_data())
    for d,m,a,l in tqdm(mydataset.generate_data()):
        print(d.shape)
        print(m.shape)
        print(a.shape)
        print(l.shape)
        print(l[:5])
        print(np.argmax(l[:5].cpu().numpy(), 1))
        print(torch.tensor(np.isin(np.argmax(l[:5].cpu().numpy(), 1), [2], invert=True)))