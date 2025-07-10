import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

import os


class ISIC20Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.columns = [col for col in self.data_frame.columns if col not in ['image_name', 'benign_malignant', 'patient_id']]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_frame.iloc[idx]['image_name'] + '.jpg')
        label = self.data_frame.iloc[idx]['benign_malignant']

        image = Image.open(img_path).convert("RGB")

        # get metas (make sure to format the df before)
        # metas = self.data_frame.iloc[idx][self.columns].values # fix later
        metas = [1, 0, 0, 1]

        if self.transform:
            image = self.transform(image)

        return image, metas, label


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=512, get_latent=False):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.get_latent = get_latent

        # 256x256 -> 16x16
        self.encoder_conv = nn.Sequential(
            # in_channels, out_channels, _kernel_size, stride, padding
            # kernel size 4, stride 2 and padding 1 for halfing the dimension:

            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 32 -> 16
            nn.ReLU(),
        )

        # project to latent space
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim)
        )

        # project back from latent space
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU()
        )

        # latent_space -> 256x256
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64 -> 128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128 -> 256
            nn.Sigmoid(),  # [0, 1] output
        )

    def forward(self, x):
        # encoder
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)

        # toma de graça ae Jaubert 
        if self.get_latent:
            return nn.Flatten(x)
        
        # decoder
        x = self.decoder_fc(x)
        x = x.view(-1, 256, 16, 16)
        x = self.decoder_conv(x)
        return x