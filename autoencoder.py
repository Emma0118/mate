import torch
import torch.nn as nn
import torchvision.transforms as transforms

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.ec = nn.Linear(256, 512)
        self.dc = nn.Linear(512, 256)
        

    def forward(self, x):
        x = self.ec(x)
        x = self.dc(x)

        return x




# net = Autoencoder()

# W = net.ec.weight
# bias = net.ec.bias
# print('Weight = ', W)
# print('bias = ', bias)

# for name, params in net.named_parameters():

#     print('name:', name, 'paramters: ', params)

                            

  
