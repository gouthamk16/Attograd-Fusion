import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from unet import Unet
import matplotlib.pyplot as plt
import numpy as np
from noise_scheduler import NoiseScheduler

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

## Function to display a tensor as an image
def display_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    plt.imshow(np.transpose(tensor, (1, 2, 0)))
    plt.show()

# Loading and processing the dataset

transform = transforms.Compose([
    transforms.Resize(size=(572, 572)),
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

    
## Testing the implementation of the Unet 
image_sample, label_sample = next(iter(trainloader))
print(image_sample.shape)
model = Unet()
output = model(image_sample[0])
print(output.shape)


noise = NoiseScheduler(timesteps=1000, beta_start=0.1, beta_end=0.2)
x_t, noise = noise.forward_diffusion(image_sample[0], 999)
print(x_t.shape)
print(noise.shape)
display_image(x_t)
display_image(x_t)