import torch
from utils import *
from torchvision import transforms

from torch.utils.data.dataset import Dataset as dataset
from torch.utils.data import DataLoader


src_domains, (X_test, y_test) = load_rotated_MNIST()






X_list = []

for d in range(len(src_domains)):
    X, y = src_domains[d]
    X_list.append(X)
X_list = np.array(X_list)
# print(X_list.shape) #(5, 1000, 256)



X_in, X_outs = construct_pair(X_list)

# print('X_in.shape', X_in.shape) (5000, 256)
# print('X_outs = ', len(X_outs)) # 5

# print('X_outs[0].shape ', X_outs[0].shape) (5000, 256)

normed = (X_in - X_in.mean()) / X_in.std()
normed = (normed - normed.mean(axis=0)) / normed.std(axis=0)



print('X_innormed_Norm.shape', normed.shape)



class Dataset(dataset):
    def __init__(self, train=True, dom=0):
        super(Dataset, self).__init__()
        self.dom = dom
        if train:
            self.inputs = normed  # inputs of all domains
            self.outs = X_outs[self.dom]  # the matrix of replicated data sets taken from the lth domain
        else:
            (self.inputs, self.outs) = (X_test, y_test)

        # self.images = self.images.reshape(-1, 1, 256)
    def __getitem__(self, index):
        input = self.inputs[index]
        output = self.outs[index]
        
        return input, output

    def __len__(self):
        return len(self.inputs)

    pass

data = Dataset()

trainloader = DataLoader(data, batch_size=4,shuffle=True)
dataiter = iter(trainloader)
input, output = dataiter.next()

# print(input.shape)
# print(len(output)) # (4, 256)


