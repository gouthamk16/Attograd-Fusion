import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

## Autoencoder -> Compressing image features to a smaller latent space using KL regularization 
## Generating the image from the trained latent representation - unets?
## Unet architecture - 2D conv layers -> time conditional unet (used by the latent diffusion model) - https://arxiv.org/pdf/2112.10752
"""
1. Skip connections
2. Concat layers
3. Cross Attention
"""
# transformer? (unet)

## Conditional model - conditional denoising autoencoders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading and processing the dataset

transform = transforms.Compose([
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010) # Mean and standard deviation used in the proessing of image net
    )
])

trainset = CIFAR10(
    root = "./data",
    train = True,
    download = True,
    transform = transform,
    target_transform = None
)
testset = CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transform,
    target_transform = None
)

trainloader = DataLoader(
    dataset = trainset,
    batch_size = 32,
    shuffle = True
)
testloader = DataLoader(
    dataset = testset,
    batch_size = 32,
    shuffle = True
)

## Creating the encoder model for the Unet
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxPool(x)
        x = self.layer2(x)
        x = self.maxPool(x)
        x = self.layer3(x)
        x = self.maxPool(x)
        x = self.layer4(x)
        x = self.maxPool(x)
        x = self.layer5(x)
    
## Testing the minor implementation of the encoder -> not yet complete
image_sample, label_sample = next(iter(trainloader))
print(image_sample.shape)
model = Encoder()
output = model(image_sample[0])
print(output.shape)