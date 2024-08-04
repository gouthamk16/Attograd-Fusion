import torch.nn as nn

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
        print("Shape after encoding: ", x.shape)
        return x
    
## Creating the decoder model for the Unet
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=(2, 2), stride=(2, 2))
        self.upconv2 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
        self.upconv3 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
        self.layer1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=(1, 1))
        )
    def forward(self, x):
        x = self.upconv1(x)
        x = self.layer1(x)
        x = self.upconv2(x)
        x = self.layer2(x)
        x = self.upconv3(x)
        x = self.layer3(x)
        x = self.upconv4(x)
        x = self.layer4(x)
        return x
    
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x