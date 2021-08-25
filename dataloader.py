import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from glob import glob
#from skimage.transform import resize

class cyclegan_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, direction='A2B', transform=None):
        self.data_dir = data_dir
        self.data_dir_A = os.path.join(data_dir, 'A')
        self.data_dir_B = os.path.join(data_dir, 'B')
        self.transform = transform
        self.direction = direction

        lst_A = sorted(list(glob(self.data_dir_A + '\*.jpg')))
        lst_B = sorted(list(glob(self.data_dir_B + '\*.jpg')))

        self.names = (lst_A, lst_B)

    def __len__(self):
        return len(self.names[0])

    def __getitem__(self, idx):
        dataA = plt.imread(self.names[0][idx]).squeeze()
        dataB = plt.imread(self.names[1][idx]).squeeze()
        sizeA = dataA.shape
        sizeB = dataB.shape
        if (sizeA != (256, 256, 3)):
            print(self.names[0][idx])
        if (sizeB != (256, 256, 3)):
            print(self.names[1][idx])

        if dataA.ndim == 2:
            dataA = dataA[:, :, np.newaxis]
        if dataA.dtype == np.uint8:
            dataA = dataA / 255.0

        if dataB.ndim == 2:
            dataB = dataB[:, :, np.newaxis]
        if dataB.dtype == np.uint8:
            dataB = dataB / 255.0

        if self.direction == 'A2B':  
            data = {'A': dataA, 'B': dataB}
        else :
            data = {'A': dataB, 'B': dataA}
        
        if self.transform: 
            data = self.transform(data)
        
        return data

# Transform 구현
class ToTensor(object):
    def __call__(self, data):
        A, B = data['A'], data['B']

        A = A.transpose((2, 0, 1)).astype(np.float32)
        B = B.transpose((2, 0, 1)).astype(np.float32)

        data = {'A': torch.from_numpy(A), 'B': torch.from_numpy(B)}
        return data

class ToNumpy(object):
    def __call__(self, data):
        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

class Normalization(object):
    def __call__(self, data):
        # [0, 1] to [-1, 1]
        A, B = data['A'], data['B']
        A = 2*A - 1
        B = 2*B - 1

        data = {'A': A, 'B': B}
        return data

class Denormalize(object):
    def __call__(self, data):
        # [-1, 1] to [0, 1]
        A, B = data['A'], data['B']
        A = (A+1) / 2
        B = (B+1) / 2

        data = {'A': A, 'B': B}
        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}
        return data

class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        label, input = data['label'], data['input']
        h, w = data['label'].shape[:2]
        nh, nw = self.shape

        dh = np.random.randint(0, h - nh)
        dw = np.random.randint(0, w - nw)
        rh = np.arange(dh, dh + nh, 1)[:, np.newaxis]
        rw = np.arange(dw, dw + nw, 1)

        label = label[rh, rw]
        input = input[rh, rw]
        data = {'label': label, 'input': input}
        return data

"""
class Resize(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        label, input = data['label'], data['input']
        label = resize(label, (self.shape[0], self.shape[1], self.shape[2]))
        input = resize(input, (self.shape[0], self.shape[1], self.shape[2]))

        data = {'label': label, 'input': input}
        return data
        """