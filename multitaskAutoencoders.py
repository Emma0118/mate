import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from autoencoder import Autoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_epochs = 500
batch_size = 128
learning_rate = 0.001


model = Autoencoder().to(device)
criterion = nn.MSELoss()







class MultitaskAutoencoder(nn.Module):
    def __init__(self):
        super(MultitaskAutoencoder,self).__init__()
        
        self.ec = nn.Linear(256, 512)

        self.dc1= nn.Linear(512, 256)
        self.dc2= nn.Linear(512, 256)
        self.dc3 = nn.Linear(512, 256)
        self.dc4= nn.Linear(512, 256)
        self.dc5= nn.Linear(512, 256)
    

    def forward(self, x, dom=0):
        x = self.ec(x)
        out1 = self.dc1(x)
        out2 = self.dc2(x)
        out3 =self.dc3(x)
        out4 = self.dc4(x)
        out5 = self.dc5(x)

        return out1
       

        

multitaskAE = MultitaskAutoencoder().to(device)
# print(multitaskAE)

optimizer = torch.optim.Adam(multitaskAE.parameters(), lr=learning_rate, weight_decay=1e-5)


# test_data = torch.randn(3, 1, 256).to(device)
# out = multitaskAE(test_data)

# print(out.shape)


def train_AE():
    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(100):
        train_loss = 0.0

        for input, out in dataloader:
            input = input.float().to(device)

            # print('input.shape = ', input.shape)
            dom_out = out.float().to(device)
            # print('dom_out.shape = ', dom_out.shape)

            output = multitaskAE(input)
            # print('predicted output.shape = ', output.shape)

            loss = criterion(output, dom_out)
            loss.backward()
            optimizer.zero_grad()

            train_loss += loss.item()

        print('===> Loss: ', train_loss/len(dataloader))



train_AE()




# dataset = Dataset()
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for epoch in range(100):
#     losses = []
#     for data in dataloader:
#         img = data[0].float().to(device)
#         out = []
#         for i in data[1]:
#             out.append(i.float().to(device))

#         outputs = multitaskAE(img)  # out1, out2, out3, out4, out5

#         for i in range(len(outputs)):
#             loss = criterion(outputs[i], out[i])
#             losses.append(loss)
#         loss = sum(losses)
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         optimizer.step()

#     print('===> Loss: ', loss)
