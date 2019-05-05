'''
Reference: https://github.com/pytorch/examples/blob/master/vae/main.py, 
           https://github.com/hwalsuklee/tensorflow-mnist-VAE
'''

import argparse
import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


# --- parsing and configuration --- #
parser = argparse.ArgumentParser(
    description="PyTorch implementation of VAE for MNIST")
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=60,
                    help='number of epochs to train (default: 60)')
parser.add_argument('--z-dim', type=int, default=2,
                    help='dimension of hidden variable Z (default: 2)')
parser.add_argument('--log-interval', type=int, default=100,
                    help='interval between logs about training status (default: 100)')
parser.add_argument('--learning-rate', type=int, default=1e-3,
                    help='learning rate for Adam optimizer (default: 1e-3)')
parser.add_argument('--prr', type=bool, default=True,
                    help='Boolean for plot-reproduce-result (default: True')
parser.add_argument('--prr-z1-range', type=int, default=2,
                    help='z1 range for plot-reproduce-result (default: 2)')
parser.add_argument('--prr-z2-range', type=int, default=2,
                    help='z2 range for plot-reproduce-result (default: 2)')
parser.add_argument('--prr-z1-interval', type=int, default=0.1,
                    help='interval of z1 for plot-reproduce-result (default: 0.1)')
parser.add_argument('--prr-z2-interval', type=int, default=0.1,
                    help='interval of z2 for plot-reproduce-result (default: 0.1)')

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LOG_INTERVAL = args.log_interval
Z_DIM = args.z_dim
LEARNING_RATE = args.learning_rate
PRR = args.prr
Z1_RANGE = args.prr_z1_range
Z2_RANGE = args.prr_z2_range
Z1_INTERVAL = args.prr_z1_interval
Z2_INTERVAL = args.prr_z2_interval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- data loading --- #
train_data = datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST('./data', train=False,
                           transform=transforms.ToTensor())

# pin memory provides improved transfer speed
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=BATCH_SIZE, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=BATCH_SIZE, shuffle=True, **kwargs)


# --- defines the model and the optimizer --- #
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, Z_DIM)  # fc21 for mean of Z
        self.fc22 = nn.Linear(500, Z_DIM)  # fc22 for log variance of Z
        self.fc3 = nn.Linear(Z_DIM, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        # I guess the reason for using logvar instead of std or var is that
        # the output of fc22 can be negative value (std and var should be positive)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # x: [batch size, 1, 28,28] -> x: [batch size, 784]
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- defines the loss function --- #
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    return BCE + KLD


# --- train and test --- #
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        # data: [batch size, 1, 28, 28]
        # label: [batch size] -> we don't use
        optimizer.zero_grad()
        data = data.to(device)
        recon_data, mu, logvar = model(data)
        loss = loss_function(recon_data, data, mu, logvar)
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader),
                cur_loss/len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)
    ))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, mu, logvar = model(data)
            cur_loss = loss_function(recon_data, data, mu, logvar).item()
            test_loss += cur_loss
            if batch_idx == 0:
                # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
                num_samples = min(BATCH_SIZE, 8)
                comparison = torch.cat(
                    [data[:num_samples], recon_data.view(BATCH_SIZE, 1, 28, 28)[:num_samples]]).cpu()
                save_generated_img(
                    comparison, 'reconstruction', epoch, num_samples)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# --- etc. funtions --- #
def save_generated_img(image, name, epoch, nrow=8):
    if not os.path.exists('results'):
        os.makedirs('results')

    if epoch % 5 == 0:
        save_path = 'results/'+name+'_'+str(epoch)+'.png'
        save_image(image, save_path, nrow=nrow)


def sample_from_model(epoch):
    with torch.no_grad():
        # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
        sample = torch.randn(64, Z_DIM).to(device)
        sample = model.decode(sample).cpu().view(64, 1, 28, 28)
        save_generated_img(sample, 'sample', epoch)


def plot_along_axis(epoch):
    z1 = torch.arange(-Z1_RANGE, Z1_RANGE, Z1_INTERVAL).to(device)
    z2 = torch.arange(-Z2_RANGE, Z2_RANGE, Z2_INTERVAL).to(device)
    num_z1 = z1.shape[0]
    num_z2 = z2.shape[0]
    num_z = num_z1 * num_z2

    sample = torch.zeros(num_z, 2).to(device)

    for i in range(num_z1):
        for j in range(num_z2):
            idx = i * num_z2 + j
            sample[idx][0] = z1[i]
            sample[idx][1] = z2[j]

    sample = model.decode(sample).cpu().view(num_z, 1, 28, 28)
    save_generated_img(sample, 'plot_along_z1_and_z2_axis_', epoch, num_z1)


# --- main function --- #
if __name__ == '__main__':
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        sample_from_model(epoch)

        if PRR:
            plot_along_axis(epoch)
